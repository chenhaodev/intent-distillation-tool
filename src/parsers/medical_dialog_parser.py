"""
Medical Dialog Parser
Parses medical conversation transcripts from MedVAL-Bench format
"""
import re
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime


class MedicalDialogParser:
    """Parse medical dialogues from CSV format into structured conversations"""

    def __init__(self):
        self.conversation_counter = 0

    def parse_csv(self, csv_path: str, task_filter: str = "dialogue2note") -> List[Dict[str, Any]]:
        """
        Parse CSV file and extract medical dialogues

        Args:
            csv_path: Path to CSV file
            task_filter: Task type to filter (default: dialogue2note)

        Returns:
            List of structured conversation dictionaries
        """
        df = pd.read_csv(csv_path)

        # Filter by task type if specified
        if task_filter:
            df = df[df['task'] == task_filter]

        conversations = []
        for idx, row in df.iterrows():
            conv = self._parse_dialogue_text(row['input'], row['id'])
            if conv:
                conversations.append(conv)

        return conversations

    def _parse_dialogue_text(self, dialogue_text: str, dialogue_id: int) -> Dict[str, Any]:
        """
        Parse dialogue text into structured turns

        Args:
            dialogue_text: Raw dialogue text with [doctor]/[patient] markers
            dialogue_id: Original dialogue ID from dataset

        Returns:
            Structured conversation dictionary
        """
        # Split by speaker markers
        # Pattern: [doctor] or [patient]
        pattern = r'\[(doctor|patient)\]\s*'

        # Find all speaker markers and their positions
        splits = re.split(pattern, dialogue_text)

        # Remove empty strings and reconstruct turns
        turns = []
        current_speaker = None

        for i, segment in enumerate(splits):
            segment = segment.strip()
            if not segment:
                continue

            if segment in ['doctor', 'patient']:
                current_speaker = segment
            elif current_speaker:
                # Map to standard roles
                role = 'assistant' if current_speaker == 'doctor' else 'user'
                turns.append({
                    'role': role,
                    'content': segment,
                    'original_speaker': current_speaker
                })
                current_speaker = None

        if not turns:
            return None

        # Generate conversation ID
        self.conversation_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d")

        return {
            'conversation_id': f'medval_{dialogue_id}_{timestamp}',
            'original_id': int(dialogue_id),
            'source': 'medval_bench',
            'domain': 'medical',
            'task': 'dialogue2note',
            'turns': turns,
            'num_turns': len(turns)
        }

    def get_statistics(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about parsed conversations"""
        if not conversations:
            return {}

        total_turns = sum(c['num_turns'] for c in conversations)
        user_turns = sum(len([t for t in c['turns'] if t['role'] == 'user']) for c in conversations)

        return {
            'total_conversations': len(conversations),
            'total_turns': total_turns,
            'user_turns': user_turns,
            'assistant_turns': total_turns - user_turns,
            'avg_turns_per_conversation': total_turns / len(conversations) if conversations else 0,
            'min_turns': min(c['num_turns'] for c in conversations),
            'max_turns': max(c['num_turns'] for c in conversations)
        }
