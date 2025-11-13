"""
Dataset exporters for SLM training
Supports Alpaca, ShareGPT, and custom formats
"""
import json
import csv
from typing import List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DatasetExporter:
    """Export classification results to SLM training formats"""

    @staticmethod
    def export_to_alpaca(
        results: List[Dict[str, Any]],
        output_path: str,
        system_prompt: str = ""
    ) -> None:
        """
        Export to Alpaca format

        Args:
            results: List of classification results or distillation results
            output_path: Output file path
            system_prompt: Optional system prompt
        """
        alpaca_data = []

        for result in results:
            # For distillation results (question + intent)
            if "question" in result and "intent" in result:
                instruction = system_prompt or "Classify the intent of the following user query."
                alpaca_data.append({
                    "instruction": instruction,
                    "input": result["question"],
                    "output": result["intent"]
                })

            # For classification results (input + intent + confidence)
            elif "input" in result and "intent" in result:
                alpaca_data.append({
                    "instruction": "Classify the intent of the following user input.",
                    "input": result["input"],
                    "output": f"Intent: {result['intent']}\nConfidence: {result['confidence']}\nReasoning: {result['reasoning']}",
                    "system": system_prompt
                })

            # For image classification
            elif "category" in result:
                alpaca_data.append({
                    "instruction": "Classify the category of the image.",
                    "input": f"Image: {result['image_path']}",
                    "output": f"Category: {result['category']}\nConfidence: {result['confidence']}\nReasoning: {result['reasoning']}\nScene: {result.get('scene_description', '')}",
                    "system": system_prompt
                })

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported {len(alpaca_data)} samples to Alpaca format: {output_path}")

    @staticmethod
    def export_to_sharegpt(
        results: List[Dict[str, Any]],
        output_path: str,
        system_prompt: str = ""
    ) -> None:
        """
        Export to ShareGPT format

        Args:
            results: List of classification results or distillation results
            output_path: Output file path
            system_prompt: Optional system prompt
        """
        sharegpt_data = []

        for result in results:
            messages = []

            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            # For distillation results (question + intent)
            if "question" in result and "intent" in result:
                messages.append({
                    "role": "user",
                    "content": result["question"]
                })
                messages.append({
                    "role": "assistant",
                    "content": result["intent"]
                })

            # For classification results (input + intent + confidence)
            elif "input" in result and "intent" in result:
                messages.append({
                    "role": "user",
                    "content": f"Classify the intent: {result['input']}"
                })
                messages.append({
                    "role": "assistant",
                    "content": f"The intent is '{result['intent']}' with confidence {result['confidence']}. {result['reasoning']}"
                })

            # For image classification
            elif "category" in result:
                messages.append({
                    "role": "user",
                    "content": f"Classify this image: {result['image_path']}"
                })
                messages.append({
                    "role": "assistant",
                    "content": f"The category is '{result['category']}' with confidence {result['confidence']}. {result['reasoning']}"
                })

            sharegpt_data.append({"messages": messages})

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported {len(sharegpt_data)} samples to ShareGPT format: {output_path}")

    @staticmethod
    def export_to_json(results: List[Dict[str, Any]], output_path: str) -> None:
        """Export to JSON format"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Exported {len(results)} results to JSON: {output_path}")

    @staticmethod
    def export_to_jsonl(results: List[Dict[str, Any]], output_path: str) -> None:
        """Export to JSONL format"""
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        logger.info(f"Exported {len(results)} results to JSONL: {output_path}")

    @staticmethod
    def export_to_csv(results: List[Dict[str, Any]], output_path: str) -> None:
        """Export to CSV format"""
        if not results:
            logger.warning("No results to export")
            return

        # Determine fields based on first result
        first_result = results[0]
        fieldnames = list(first_result.keys())

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                # Convert lists to strings for CSV
                row = {}
                for key, value in result.items():
                    if isinstance(value, (list, dict)):
                        row[key] = json.dumps(value, ensure_ascii=False)
                    else:
                        row[key] = value
                writer.writerow(row)

        logger.info(f"Exported {len(results)} results to CSV: {output_path}")

    @staticmethod
    def export_conversations_to_alpaca(
        conversations: List[Dict[str, Any]],
        output_path: str,
        mode: str = "intent-classification",
        system_prompt: str = ""
    ) -> None:
        """
        Export conversations to Alpaca format with different modes

        Args:
            conversations: List of conversation dicts
            output_path: Output file path
            mode: "intent-classification" or "conversation"
            system_prompt: Optional system prompt
        """
        alpaca_data = []

        for conv in conversations:
            if mode == "intent-classification":
                # Extract each user turn as a separate intent classification sample
                turns = conv.get("turns", [])
                for i, turn in enumerate(turns):
                    if turn["role"] == "user" and "intent" in turn:
                        # Build context from previous turns
                        context_turns = turns[:i]
                        if context_turns:
                            context = "\n".join([f"{t['role'].capitalize()}: {t['content']}" for t in context_turns[-4:]])
                            input_text = f"Previous conversation:\n{context}\n\nCurrent message:\n{turn['content']}"
                            instruction = "Given the conversation context, classify the intent of the user's current message."
                        else:
                            input_text = turn['content']
                            instruction = system_prompt or "Classify the intent of the following user query."

                        alpaca_data.append({
                            "instruction": instruction,
                            "input": input_text,
                            "output": turn['intent'],
                            "metadata": {
                                "conversation_id": conv.get("conversation_id"),
                                "turn": turn.get("turn"),
                                "has_context": bool(context_turns)
                            }
                        })

            elif mode == "conversation":
                # Export full conversation as a single sample
                turns = conv.get("turns", [])
                if not turns:
                    continue

                first_user_msg = next((t["content"] for t in turns if t["role"] == "user"), "")
                full_conv = "\n".join([f"{t['role'].capitalize()}: {t['content']}" for t in turns])

                alpaca_data.append({
                    "instruction": system_prompt or "You are a helpful assistant. Engage in a multi-turn conversation.",
                    "input": first_user_msg,
                    "output": full_conv,
                    "metadata": {
                        "conversation_id": conv.get("conversation_id"),
                        "primary_intent": conv.get("primary_intent"),
                        "all_intents": conv.get("all_intents", [])
                    }
                })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported {len(alpaca_data)} conversation samples to Alpaca format: {output_path}")

    @staticmethod
    def export_conversations_to_sharegpt(
        conversations: List[Dict[str, Any]],
        output_path: str,
        system_prompt: str = ""
    ) -> None:
        """
        Export conversations to ShareGPT format (native multi-turn format)

        Args:
            conversations: List of conversation dicts
            output_path: Output file path
            system_prompt: Optional system prompt
        """
        sharegpt_data = []

        for conv in conversations:
            turns = conv.get("turns", [])
            messages = []

            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            for turn in turns:
                messages.append({
                    "role": turn["role"],
                    "content": turn["content"]
                })

            sharegpt_data.append({
                "messages": messages,
                "metadata": {
                    "conversation_id": conv.get("conversation_id"),
                    "primary_intent": conv.get("primary_intent"),
                    "all_intents": conv.get("all_intents", []),
                    "num_turns": conv.get("num_turns", 0)
                }
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported {len(sharegpt_data)} conversations to ShareGPT format: {output_path}")

    @classmethod
    def export(
        cls,
        results: List[Dict[str, Any]],
        output_path: str,
        format: str = "jsonl",
        **kwargs
    ) -> None:
        """
        Export results to specified format

        Args:
            results: List of classification results, distillation results, or conversations
            output_path: Output file path
            format: Export format (json, jsonl, csv, alpaca, sharegpt)
            **kwargs: Additional arguments (e.g., system_prompt, mode)
        """
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Check if this is conversation data
        is_conversation = (
            results and
            isinstance(results[0], dict) and
            "conversation_id" in results[0] and
            "turns" in results[0]
        )

        mode = kwargs.get("mode") or "intent-classification"

        if format == "json":
            cls.export_to_json(results, output_path)
        elif format == "jsonl":
            cls.export_to_jsonl(results, output_path)
        elif format == "csv":
            cls.export_to_csv(results, output_path)
        elif format == "alpaca":
            if is_conversation:
                cls.export_conversations_to_alpaca(results, output_path, mode, kwargs.get("system_prompt", ""))
            else:
                cls.export_to_alpaca(results, output_path, kwargs.get("system_prompt", ""))
        elif format == "sharegpt":
            if is_conversation:
                cls.export_conversations_to_sharegpt(results, output_path, kwargs.get("system_prompt", ""))
            else:
                cls.export_to_sharegpt(results, output_path, kwargs.get("system_prompt", ""))
        else:
            raise ValueError(f"Unsupported export format: {format}")
