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
            results: List of classification results
            output_path: Output file path
            format: Export format (json, jsonl, csv, alpaca, sharegpt)
            **kwargs: Additional arguments (e.g., system_prompt for alpaca/sharegpt)
        """
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            cls.export_to_json(results, output_path)
        elif format == "jsonl":
            cls.export_to_jsonl(results, output_path)
        elif format == "csv":
            cls.export_to_csv(results, output_path)
        elif format == "alpaca":
            cls.export_to_alpaca(results, output_path, kwargs.get("system_prompt", ""))
        elif format == "sharegpt":
            cls.export_to_sharegpt(results, output_path, kwargs.get("system_prompt", ""))
        else:
            raise ValueError(f"Unsupported export format: {format}")
