"""
Medical Intent Tagger
Generate intent tags for medical conversations using LLM
"""
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.llm.client import LLMClient
from src.llm.prompts.medical_intent_tagging import get_medical_intent_tagging_prompt
from src.distillers.intent_tag_distiller import IntentNode
from src.utils.taxonomy_utils import export_taxonomy_text

logger = logging.getLogger(__name__)


class MedicalIntentTagger:
    """Generate intent tags for medical conversations"""

    def __init__(self, llm_client: LLMClient, language: str = "en", taxonomy: IntentNode = None):
        """
        Initialize medical intent tagger

        Args:
            llm_client: LLM client for generating intent tags
            language: Language code (en/zh)
            taxonomy: Optional intent taxonomy tree to guide tagging
        """
        self.llm_client = llm_client
        self.language = language
        self.taxonomy = taxonomy

        # Build taxonomy text if provided
        taxonomy_text = None
        if taxonomy:
            taxonomy_text = export_taxonomy_text(taxonomy)

        self.prompt_template = get_medical_intent_tagging_prompt(language, taxonomy_text)

    def tag_conversation(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate intent tags for a single conversation

        Args:
            conversation: Conversation dictionary with 'turns' field

        Returns:
            Conversation with intent tags added
        """
        # Format conversation for prompt
        conversation_text = self._format_conversation(conversation['turns'])

        # Generate intent tags using LLM
        prompt = self.prompt_template['user'].format(conversation=conversation_text)

        try:
            response = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": self.prompt_template['system']},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )

            # Parse LLM response (extract text from response dict)
            response_text = response.get("text", response) if isinstance(response, dict) else response
            intent_data = self._parse_llm_response(response_text)

            # Merge intent data with conversation
            tagged_conversation = self._merge_intents(conversation, intent_data)

            return tagged_conversation

        except Exception as e:
            logger.error(f"Error tagging conversation {conversation.get('conversation_id')}: {e}")
            # Return conversation with empty intents on error
            return self._create_fallback_conversation(conversation)

    def tag_conversations_batch(
        self,
        conversations: List[Dict[str, Any]],
        batch_size: int = 1,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Tag multiple conversations in batches

        Args:
            conversations: List of conversation dictionaries
            batch_size: Number of conversations to process at once (currently only 1 supported)
            progress_callback: Optional callback function for progress updates

        Returns:
            List of tagged conversations
        """
        tagged_conversations = []

        for i, conversation in enumerate(conversations):
            try:
                tagged = self.tag_conversation(conversation)
                tagged_conversations.append(tagged)

                if progress_callback:
                    progress_callback(i + 1, len(conversations))

            except Exception as e:
                logger.error(f"Error processing conversation {i}: {e}")
                # Add fallback on error
                tagged_conversations.append(self._create_fallback_conversation(conversation))

        return tagged_conversations

    def _format_conversation(self, turns: List[Dict[str, str]]) -> str:
        """Format conversation turns for LLM prompt"""
        formatted_turns = []

        for i, turn in enumerate(turns):
            role_label = "Patient" if turn['role'] == 'user' else "Doctor"
            formatted_turns.append(f"[{role_label}] {turn['content']}")

        return "\n\n".join(formatted_turns)

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response"""
        try:
            # Try to parse as JSON
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

            logger.warning("Failed to parse LLM response as JSON")
            return {
                'user_turns': [],
                'primary_intent': 'Unknown',
                'all_intents': [],
                'conversation_summary': ''
            }

    def _merge_intents(
        self,
        conversation: Dict[str, Any],
        intent_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge intent data into conversation structure"""

        # Create a map of turn_index to intent info
        intent_map = {
            item['turn_index']: item
            for item in intent_data.get('user_turns', [])
        }

        # Add intents to user turns
        tagged_turns = []
        user_turn_count = 0

        for i, turn in enumerate(conversation['turns']):
            tagged_turn = turn.copy()

            if turn['role'] == 'user':
                intent_info = intent_map.get(i, intent_map.get(user_turn_count, {}))
                tagged_turn['intent'] = intent_info.get('intent', 'Unknown')
                tagged_turn['intent_path'] = intent_info.get('intent_path', 'Unknown')
                tagged_turn['intent_reasoning'] = intent_info.get('reasoning', '')
                user_turn_count += 1

            tagged_turns.append(tagged_turn)

        # Create tagged conversation
        tagged_conversation = conversation.copy()
        tagged_conversation['turns'] = tagged_turns
        tagged_conversation['primary_intent'] = intent_data.get('primary_intent', 'Unknown')
        tagged_conversation['all_intents'] = intent_data.get('all_intents', [])
        tagged_conversation['conversation_summary'] = intent_data.get('conversation_summary', '')
        tagged_conversation['tagged_at'] = datetime.now().isoformat()

        return tagged_conversation

    def _create_fallback_conversation(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback conversation with empty intents"""
        fallback = conversation.copy()
        fallback['turns'] = [
            {**turn, 'intent': 'Unknown', 'intent_path': 'Unknown'}
            if turn['role'] == 'user' else turn
            for turn in conversation['turns']
        ]
        fallback['primary_intent'] = 'Unknown'
        fallback['all_intents'] = []
        fallback['conversation_summary'] = ''
        fallback['tagged_at'] = datetime.now().isoformat()
        fallback['tagging_error'] = True

        return fallback
