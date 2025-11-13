"""
Intent Conversation Distiller
Generates multi-turn conversations with intent transitions for training conversational AI
"""
import json
import logging
import random
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.llm.prompts.distill_conversations import (
    build_assistant_reply_prompt,
    build_next_question_prompt
)

logger = logging.getLogger(__name__)


class IntentConversationDistiller:
    """Generate multi-turn conversations with intent awareness"""

    def __init__(self, llm_client, language: str = "en"):
        """
        Initialize the conversation distiller

        Args:
            llm_client: LLM client for generating conversations
            language: Language for generation (en/zh)
        """
        self.llm_client = llm_client
        self.language = language

        # Default scenario descriptions
        self.default_scenarios = {
            "en": "A helpful customer support conversation where users seek assistance and the assistant provides professional guidance.",
            "zh": "一个有帮助的客户支持对话，用户寻求帮助，助手提供专业指导。"
        }

        # Default role names
        self.default_roles = {
            "en": {"user": "User", "assistant": "Assistant"},
            "zh": {"user": "用户", "assistant": "助手"}
        }

    def distill_conversation(
        self,
        intent_node,
        turns: int = 4,
        transition_rate: float = 0.3,
        scenario: Optional[str] = None,
        role_user: Optional[str] = None,
        role_assistant: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a single multi-turn conversation for an intent

        Args:
            intent_node: IntentNode from taxonomy
            turns: Number of turns in conversation (user+assistant pairs)
            transition_rate: Probability of transitioning to related intent (0-1)
            scenario: Custom conversation scenario description
            role_user: Custom user role name
            role_assistant: Custom assistant role name

        Returns:
            Dictionary containing the conversation data
        """
        # Set defaults
        scenario = scenario or self.default_scenarios[self.language]
        roles = self.default_roles[self.language]
        role_user = role_user or roles["user"]
        role_assistant = role_assistant or roles["assistant"]

        # Get related intents (siblings or parent's other children)
        related_intents = self._get_related_intents(intent_node)

        conversation = []
        current_intent = intent_node
        all_intents = [intent_node.name]
        transition_points = []

        # Generate initial user question
        logger.info(f"Generating conversation for intent: {intent_node.full_name}")

        # First question about the primary intent
        first_question = self._generate_initial_question(intent_node)
        conversation.append({
            "role": "user",
            "content": first_question,
            "intent": intent_node.name,
            "turn": 1
        })

        # Generate conversation turns
        for turn_idx in range(1, turns + 1):
            # Generate assistant response
            conversation_history = self._format_conversation_history(conversation)

            assistant_reply = self._generate_assistant_reply(
                scenario=scenario,
                role_user=role_user,
                role_assistant=role_assistant,
                current_intent=current_intent.name,
                intent_path=current_intent.path if hasattr(current_intent, 'path') else current_intent.name,
                conversation_history=conversation_history,
                current_turn=turn_idx,
                total_turns=turns
            )

            conversation.append({
                "role": "assistant",
                "content": assistant_reply,
                "turn": turn_idx + 1
            })

            # Generate next user question (if not last turn)
            if turn_idx < turns:
                # Decide if we should transition to a related intent
                should_transition = (
                    random.random() < transition_rate and
                    related_intents and
                    turn_idx > 1  # Don't transition too early
                )

                if should_transition:
                    # Transition to a related intent
                    current_intent = random.choice(related_intents)
                    if current_intent.name not in all_intents:
                        all_intents.append(current_intent.name)
                        transition_points.append(len(conversation) + 1)
                    logger.info(f"Transitioning to related intent: {current_intent.name}")

                next_question = self._generate_next_question(
                    scenario=scenario,
                    role_user=role_user,
                    role_assistant=role_assistant,
                    primary_intent=intent_node.name,
                    current_intent=current_intent,
                    related_intents=related_intents,
                    conversation_history=self._format_conversation_history(conversation),
                    next_turn=len(conversation) + 1,
                    total_turns=turns * 2,
                    transition_rate=transition_rate
                )

                conversation.append({
                    "role": "user",
                    "content": next_question["content"],
                    "intent": next_question["intent"],
                    "turn": len(conversation) + 1
                })

        return {
            "conversation_id": f"conv_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
            "primary_intent": intent_node.name,
            "primary_intent_number": intent_node.number if hasattr(intent_node, 'number') else None,
            "primary_intent_path": intent_node.path if hasattr(intent_node, 'path') else intent_node.name,
            "all_intents": all_intents,
            "turns": conversation,
            "num_turns": len(conversation),
            "transition_points": transition_points,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    def distill_conversations_for_tree(
        self,
        root_node,
        conversations_per_intent: int = 5,
        turns_per_conversation: int = 4,
        transition_rate: float = 0.3,
        leaf_only: bool = True,
        scenario: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate conversations for all intents in a taxonomy tree

        Args:
            root_node: Root IntentNode of the taxonomy
            conversations_per_intent: Number of conversations to generate per intent
            turns_per_conversation: Number of turns per conversation
            transition_rate: Probability of intent transitions
            leaf_only: Only generate for leaf nodes
            scenario: Custom conversation scenario

        Returns:
            List of conversation dictionaries
        """
        from src.distillers.intent_tag_distiller import IntentTagDistiller

        # Get target intents
        tag_distiller = IntentTagDistiller(self.llm_client, self.language)
        if leaf_only:
            target_intents = tag_distiller.get_leaf_intents(root_node)
        else:
            target_intents = tag_distiller.export_flat_list(root_node)

        all_conversations = []

        logger.info(f"Generating conversations for {len(target_intents)} intents")

        for intent_node in target_intents:
            for conv_idx in range(conversations_per_intent):
                try:
                    conversation = self.distill_conversation(
                        intent_node=intent_node,
                        turns=turns_per_conversation,
                        transition_rate=transition_rate,
                        scenario=scenario
                    )
                    all_conversations.append(conversation)
                    logger.info(
                        f"Generated conversation {conv_idx + 1}/{conversations_per_intent} "
                        f"for {intent_node.full_name}"
                    )
                except Exception as e:
                    logger.error(f"Failed to generate conversation for {intent_node.full_name}: {e}")
                    continue

        logger.info(f"Total conversations generated: {len(all_conversations)}")
        return all_conversations

    def _generate_initial_question(self, intent_node) -> str:
        """Generate the first user question about an intent"""
        # Simple template-based initial question
        templates = {
            "en": [
                f"How do I {intent_node.name.lower()}?",
                f"Can you help me with {intent_node.name.lower()}?",
                f"I need assistance with {intent_node.name.lower()}",
                f"What's the process for {intent_node.name.lower()}?"
            ],
            "zh": [
                f"如何{intent_node.name}？",
                f"能帮我{intent_node.name}吗？",
                f"我需要{intent_node.name}方面的帮助",
                f"{intent_node.name}的流程是什么？"
            ]
        }
        return random.choice(templates[self.language])

    def _generate_assistant_reply(
        self,
        scenario: str,
        role_user: str,
        role_assistant: str,
        current_intent: str,
        intent_path: str,
        conversation_history: str,
        current_turn: int,
        total_turns: int
    ) -> str:
        """Generate assistant reply using LLM"""
        prompt = build_assistant_reply_prompt(
            scenario=scenario,
            role_user=role_user,
            role_assistant=role_assistant,
            current_intent=current_intent,
            intent_path=intent_path,
            conversation_history=conversation_history,
            current_turn=current_turn,
            total_turns=total_turns,
            language=self.language
        )

        try:
            response = self.llm_client.get_json_response(
                prompt=prompt,
                system_prompt="You are a helpful assistant generating natural conversation responses."
            )
            return response.get("content", "I'm happy to help with that.")
        except Exception as e:
            logger.error(f"Error generating assistant reply: {e}")
            return "I'm happy to help with that."

    def _generate_next_question(
        self,
        scenario: str,
        role_user: str,
        role_assistant: str,
        primary_intent: str,
        current_intent,
        related_intents: List,
        conversation_history: str,
        next_turn: int,
        total_turns: int,
        transition_rate: float
    ) -> Dict[str, str]:
        """Generate next user question using LLM"""
        related_intent_names = [intent.name for intent in related_intents]
        related_intents_str = ", ".join(related_intent_names) if related_intent_names else "None"

        prompt = build_next_question_prompt(
            scenario=scenario,
            role_user=role_user,
            role_assistant=role_assistant,
            primary_intent=primary_intent,
            related_intents=related_intents_str,
            intent_path=current_intent.path if hasattr(current_intent, 'path') else current_intent.name,
            conversation_history=conversation_history,
            next_turn=next_turn,
            total_turns=total_turns,
            transition_rate=transition_rate,
            language=self.language
        )

        try:
            response = self.llm_client.get_json_response(
                prompt=prompt,
                system_prompt="You are generating natural follow-up questions in a conversation."
            )
            return {
                "content": response.get("question", "Can you tell me more about that?"),
                "intent": response.get("intent", current_intent.name)
            }
        except Exception as e:
            logger.error(f"Error generating next question: {e}")
            return {
                "content": "Can you tell me more about that?",
                "intent": current_intent.name
            }

    def _get_related_intents(self, intent_node) -> List:
        """Get related intents (siblings or nearby in taxonomy)"""
        related = []

        # Get siblings (same parent)
        if hasattr(intent_node, 'parent') and intent_node.parent:
            related.extend([
                child for child in intent_node.parent.children
                if child != intent_node
            ])

        return related[:5]  # Limit to 5 related intents

    def _format_conversation_history(self, conversation: List[Dict]) -> str:
        """Format conversation history for prompt"""
        history_lines = []
        for turn in conversation:
            role = turn["role"].capitalize()
            content = turn["content"]
            history_lines.append(f"{role}: {content}")

        return "\n".join(history_lines) if history_lines else "No previous conversation"
