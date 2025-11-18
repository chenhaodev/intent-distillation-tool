"""
Medical Taxonomy Builder
Analyzes real medical conversations to build domain-specific intent taxonomy
"""
import json
import logging
import random
from typing import List, Dict, Any, Optional

from src.llm.client import LLMClient
from src.llm.prompts.medical_taxonomy_builder import get_medical_taxonomy_prompt
from src.distillers.intent_tag_distiller import IntentNode
from src.utils.taxonomy_utils import export_taxonomy_text

logger = logging.getLogger(__name__)


class MedicalTaxonomyBuilder:
    """Build intent taxonomy from real medical conversations"""

    def __init__(self, llm_client: LLMClient, language: str = "en"):
        """
        Initialize medical taxonomy builder

        Args:
            llm_client: LLM client for building taxonomy
            language: Language code (en/zh)
        """
        self.llm_client = llm_client
        self.language = language
        self.prompt_template = get_medical_taxonomy_prompt(language)

    def build_taxonomy_from_conversations(
        self,
        conversations: List[Dict[str, Any]],
        sample_size: int = 10,
        max_turns_per_conv: int = 20
    ) -> IntentNode:
        """
        Build intent taxonomy by analyzing real conversations

        Args:
            conversations: List of conversation dictionaries
            sample_size: Number of conversations to sample for analysis
            max_turns_per_conv: Maximum turns to include per conversation

        Returns:
            Root IntentNode of the taxonomy tree
        """
        # Sample conversations for analysis
        sample_conversations = self._sample_conversations(
            conversations,
            sample_size,
            max_turns_per_conv
        )

        # Format for LLM prompt
        conversation_text = self._format_conversations_for_analysis(sample_conversations)

        # Generate taxonomy using LLM
        prompt = self.prompt_template['user'].format(
            num_conversations=len(sample_conversations),
            conversation_samples=conversation_text
        )

        try:
            response = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": self.prompt_template['system']},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )

            # Parse LLM response
            response_text = response.get("text", response) if isinstance(response, dict) else response
            taxonomy_data = self._parse_taxonomy_response(response_text)

            # Convert to IntentNode tree
            root_node = self._build_intent_tree(taxonomy_data)

            logger.info(f"Built taxonomy with {self._count_nodes(root_node)} total intents")
            return root_node

        except Exception as e:
            logger.error(f"Error building taxonomy: {e}")
            # Return default medical taxonomy on error
            return self._get_default_medical_taxonomy()

    def _sample_conversations(
        self,
        conversations: List[Dict[str, Any]],
        sample_size: int,
        max_turns_per_conv: int
    ) -> List[Dict[str, Any]]:
        """Sample conversations for analysis"""
        # If we have fewer conversations than sample_size, use all
        if len(conversations) <= sample_size:
            sampled = conversations
        else:
            sampled = random.sample(conversations, sample_size)

        # Truncate turns if needed
        truncated = []
        for conv in sampled:
            truncated_conv = conv.copy()
            if len(conv['turns']) > max_turns_per_conv:
                truncated_conv['turns'] = conv['turns'][:max_turns_per_conv]
            truncated.append(truncated_conv)

        return truncated

    def _format_conversations_for_analysis(
        self,
        conversations: List[Dict[str, Any]]
    ) -> str:
        """Format conversations for LLM prompt"""
        formatted = []

        for i, conv in enumerate(conversations, 1):
            conv_text = f"Conversation {i}:\n"
            for turn in conv['turns'][:20]:  # Limit to 20 turns per conversation
                role = "Patient" if turn['role'] == 'user' else "Doctor"
                content = turn['content'][:200]  # Truncate long turns
                conv_text += f"[{role}] {content}\n"
            formatted.append(conv_text)

        return "\n---\n".join(formatted)

    def _parse_taxonomy_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response"""
        try:
            data = json.loads(response)
            return data
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            logger.warning("Failed to parse taxonomy response")
            return {"root": {"name": "Medical Consultation", "children": []}}

    def _build_intent_tree(self, taxonomy_data: Dict[str, Any]) -> IntentNode:
        """Convert taxonomy JSON to IntentNode tree"""
        root_data = taxonomy_data.get('root', {})
        return self._build_node_recursive(root_data, None, [])

    def _build_node_recursive(
        self,
        node_data: Dict[str, Any],
        parent: Optional[IntentNode],
        path: List[str]
    ) -> IntentNode:
        """Recursively build IntentNode tree"""
        name = node_data.get('name', 'Unknown')

        # Create node (path and full_name are computed automatically from parent/name)
        node = IntentNode(name=name, parent=parent)

        # Recursively build children
        children_data = node_data.get('children', [])
        for child_data in children_data:
            child_node = self._build_node_recursive(
                child_data,
                node,
                path + [name]
            )
            node.children.append(child_node)

        return node

    def _count_nodes(self, node: IntentNode) -> int:
        """Count total nodes in tree"""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _get_default_medical_taxonomy(self) -> IntentNode:
        """Return default medical intent taxonomy"""
        if self.language == "zh":
            root = IntentNode(name="医疗咨询")

            # Level 1: Main categories
            chief_complaint = IntentNode(name="主诉", parent=root)
            medical_history = IntentNode(name="病史", parent=root)
            examination = IntentNode(name="体格检查", parent=root)
            diagnosis = IntentNode(name="诊断", parent=root)
            treatment = IntentNode(name="治疗", parent=root)

            # Level 2: Sub-categories
            chief_complaint.children.append(IntentNode(name="症状描述", parent=chief_complaint))
            chief_complaint.children.append(IntentNode(name="症状时长", parent=chief_complaint))
            chief_complaint.children.append(IntentNode(name="症状严重程度", parent=chief_complaint))

            medical_history.children.append(IntentNode(name="既往病史", parent=medical_history))
            medical_history.children.append(IntentNode(name="家族史", parent=medical_history))
            medical_history.children.append(IntentNode(name="用药史", parent=medical_history))

            examination.children.append(IntentNode(name="生命体征", parent=examination))
            examination.children.append(IntentNode(name="查体结果", parent=examination))

            diagnosis.children.append(IntentNode(name="诊断说明", parent=diagnosis))
            diagnosis.children.append(IntentNode(name="检查结果", parent=diagnosis))

            treatment.children.append(IntentNode(name="用药指导", parent=treatment))
            treatment.children.append(IntentNode(name="生活方式建议", parent=treatment))
            treatment.children.append(IntentNode(name="随访安排", parent=treatment))

            root.children = [chief_complaint, medical_history, examination, diagnosis, treatment]
        else:
            root = IntentNode(name="Medical Consultation")

            # Level 1: Main categories
            chief_complaint = IntentNode(name="Chief Complaint", parent=root)
            medical_history = IntentNode(name="Medical History", parent=root)
            examination = IntentNode(name="Physical Examination", parent=root)
            diagnosis = IntentNode(name="Diagnosis", parent=root)
            treatment = IntentNode(name="Treatment", parent=root)

            # Level 2: Sub-categories
            chief_complaint.children.append(IntentNode(name="Symptom Description", parent=chief_complaint))
            chief_complaint.children.append(IntentNode(name="Symptom Duration", parent=chief_complaint))
            chief_complaint.children.append(IntentNode(name="Symptom Severity", parent=chief_complaint))

            medical_history.children.append(IntentNode(name="Past Medical History", parent=medical_history))
            medical_history.children.append(IntentNode(name="Family History", parent=medical_history))
            medical_history.children.append(IntentNode(name="Medication History", parent=medical_history))

            examination.children.append(IntentNode(name="Vital Signs", parent=examination))
            examination.children.append(IntentNode(name="Examination Findings", parent=examination))

            diagnosis.children.append(IntentNode(name="Diagnosis Explanation", parent=diagnosis))
            diagnosis.children.append(IntentNode(name="Test Results", parent=diagnosis))

            treatment.children.append(IntentNode(name="Medication Guidance", parent=treatment))
            treatment.children.append(IntentNode(name="Lifestyle Advice", parent=treatment))
            treatment.children.append(IntentNode(name="Follow-up Scheduling", parent=treatment))

            root.children = [chief_complaint, medical_history, examination, diagnosis, treatment]

        return root
