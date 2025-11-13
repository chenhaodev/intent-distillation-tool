"""
Intent Question Distiller
Generates diverse questions for each intent through knowledge distillation
Based on easy-dataset's question distillation workflow
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..llm.client import LLMClient
from ..llm.prompts.distill_intent_questions import build_distill_intent_questions_prompt
from .intent_tag_distiller import IntentNode

logger = logging.getLogger(__name__)


class IntentQuestionDistiller:
    """Distill diverse questions for intent classification training"""

    def __init__(self, llm_client: LLMClient, language: str = "en"):
        """
        Initialize intent question distiller

        Args:
            llm_client: LLM client instance
            language: Language for prompts ('zh' or 'en')
        """
        self.llm_client = llm_client
        self.language = language

    def distill_questions(
        self,
        intent_node: IntentNode,
        count: int,
        existing_questions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Distill questions for a specific intent

        Args:
            intent_node: Intent node to generate questions for
            count: Number of questions to generate
            existing_questions: Existing questions to avoid duplicates

        Returns:
            List of question dictionaries with metadata
        """
        logger.info(f"Distilling {count} questions for intent: {intent_node.full_name}")

        # Build prompt
        prompt = build_distill_intent_questions_prompt(
            current_intent=intent_node.name,
            count=count,
            intent_path=intent_node.numbered_path,
            existing_questions=existing_questions,
            language=self.language
        )

        # Get LLM response
        try:
            response = self.llm_client.get_json_response(prompt)

            # Parse questions
            if isinstance(response, list):
                question_texts = response
            elif isinstance(response, dict) and "questions" in response:
                question_texts = response["questions"]
            else:
                raise ValueError(f"Unexpected response format: {response}")

            # Build question objects with metadata
            questions = []
            for i, question_text in enumerate(question_texts):
                question_obj = {
                    "question": question_text,
                    "intent": intent_node.name,
                    "intent_number": intent_node.number,
                    "intent_full_name": intent_node.full_name,
                    "intent_path": intent_node.path,
                    "intent_numbered_path": intent_node.numbered_path,
                    "intent_hierarchy": self._get_hierarchy(intent_node),
                    "question_index": i + 1,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
                questions.append(question_obj)

            logger.info(f"Generated {len(questions)} questions for {intent_node.full_name}")
            return questions

        except Exception as e:
            logger.error(f"Error distilling questions: {e}")
            raise

    def distill_questions_for_tree(
        self,
        root_node: IntentNode,
        questions_per_intent: int,
        leaf_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Distill questions for all intents in a tree

        Args:
            root_node: Root of intent taxonomy tree
            questions_per_intent: Number of questions per intent
            leaf_only: Only generate questions for leaf intents

        Returns:
            List of all generated question dictionaries
        """
        logger.info(f"Distilling questions for intent tree: {root_node.name}")

        all_questions = []

        if leaf_only:
            # Get leaf intents only
            target_intents = self._get_leaf_intents(root_node)
        else:
            # Get all intents
            target_intents = self._get_all_intents(root_node)

        logger.info(f"Distilling questions for {len(target_intents)} intents")

        for intent_node in target_intents:
            try:
                questions = self.distill_questions(
                    intent_node=intent_node,
                    count=questions_per_intent,
                    existing_questions=None  # Could implement global deduplication
                )
                all_questions.extend(questions)

            except Exception as e:
                logger.error(f"Failed to distill questions for {intent_node.full_name}: {e}")
                continue

        logger.info(f"Total questions distilled: {len(all_questions)}")
        return all_questions

    def _get_leaf_intents(self, node: IntentNode) -> List[IntentNode]:
        """Get all leaf intent nodes"""
        if not node.children:
            return [node]

        leaves = []
        for child in node.children:
            leaves.extend(self._get_leaf_intents(child))

        return leaves

    def _get_all_intents(self, node: IntentNode) -> List[IntentNode]:
        """Get all intent nodes in tree"""
        result = [node]
        for child in node.children:
            result.extend(self._get_all_intents(child))
        return result

    def _get_hierarchy(self, node: IntentNode) -> List[str]:
        """Get list of parent intent names up to root"""
        hierarchy = []
        current = node
        while current:
            hierarchy.insert(0, current.name)
            current = current.parent
        return hierarchy

    def augment_with_variations(
        self,
        questions: List[Dict[str, Any]],
        variations_per_question: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate variations of existing questions for data augmentation

        Args:
            questions: List of question dictionaries
            variations_per_question: Number of variations to generate per question

        Returns:
            List of augmented question dictionaries
        """
        logger.info(f"Generating {variations_per_question} variations for {len(questions)} questions")

        augmented = []

        for question_obj in questions:
            original_question = question_obj["question"]
            intent_name = question_obj["intent"]

            # Build variation prompt
            variation_prompt = f"""Generate {variations_per_question} variations of the following question that express the same intent "{intent_name}":

Original question: "{original_question}"

Requirements:
- Each variation should be a natural, different way to express the same intent
- Vary the phrasing, length, and formality
- Keep the core meaning the same

Return only a JSON array of variation strings."""

            try:
                response = self.llm_client.get_json_response(variation_prompt)

                if isinstance(response, list):
                    variations = response
                elif isinstance(response, dict) and "variations" in response:
                    variations = response["variations"]
                else:
                    logger.warning(f"Unexpected variation response format for: {original_question}")
                    continue

                # Add original question
                augmented.append(question_obj)

                # Add variations
                for i, variation in enumerate(variations):
                    variation_obj = question_obj.copy()
                    variation_obj["question"] = variation
                    variation_obj["is_variation"] = True
                    variation_obj["original_question"] = original_question
                    variation_obj["variation_index"] = i + 1
                    augmented.append(variation_obj)

            except Exception as e:
                logger.error(f"Failed to generate variations for: {original_question}: {e}")
                # Still include original
                augmented.append(question_obj)

        logger.info(f"Generated {len(augmented)} total questions (including variations)")
        return augmented
