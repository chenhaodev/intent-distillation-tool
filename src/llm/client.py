"""
LLM Client for DeepSeek and OpenRouter APIs
Based on easy-dataset's LLM client architecture
"""
import base64
import json
from typing import Dict, List, Optional, Union, Any
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM client supporting multiple providers
    Compatible with OpenAI-style APIs (DeepSeek, OpenRouter, etc.)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM client

        Args:
            config: Configuration dict with keys:
                - api_key: API key
                - base_url: API base URL
                - model: Model name
                - temperature: Sampling temperature (default: 0.7)
                - max_tokens: Maximum tokens to generate (default: 2048)
                - top_p: Nucleus sampling parameter (default: 0.9)
        """
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")
        self.model = config.get("model")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 2048)
        self.top_p = config.get("top_p", 0.9)

        if not self.api_key:
            raise ValueError("API key is required")

        # Initialize OpenAI client (compatible with DeepSeek, OpenRouter, etc.)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        logger.info(f"Initialized LLM client for {self.base_url} with model {self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Generate chat completion

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens
            response_format: Optional response format (e.g., {"type": "json_object"})

        Returns:
            Dict with 'text', 'reasoning' (if available), and 'raw' response
        """
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
                "top_p": self.top_p,
            }

            if response_format:
                params["response_format"] = response_format

            response = self.client.chat.completions.create(**params)

            # Extract text and reasoning (if available)
            choice = response.choices[0]
            text = choice.message.content or ""
            reasoning = getattr(choice.message, "reasoning_content", None)

            result = {
                "text": text,
                "reasoning": reasoning,
                "raw": response
            }

            logger.debug(f"LLM response: {text[:100]}...")
            return result

        except Exception as e:
            logger.error(f"Error in LLM chat: {e}")
            raise

    def get_response(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Get simple text response from LLM

        Args:
            prompt: User prompt (string or messages list)
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for chat()

        Returns:
            Text response
        """
        if isinstance(prompt, str):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
        else:
            messages = prompt

        result = self.chat(messages, **kwargs)
        return result["text"]

    def get_json_response(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get JSON response from LLM

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            Parsed JSON dict
        """
        # Try to use JSON mode if supported
        response_text = self.get_response(
            prompt,
            system_prompt,
            response_format={"type": "json_object"},
            **kwargs
        )

        # Parse JSON from response
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback: extract JSON from markdown code blocks
            return self._extract_json_from_text(response_text)

    def get_vision_response(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        image_base64: Optional[str] = None,
        image_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get response from vision model with image

        Args:
            prompt: Text prompt/question about the image
            image_path: Path to local image file
            image_base64: Base64-encoded image data
            image_url: URL to image
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            Dict with 'text', 'reasoning' (if available)
        """
        # Prepare image data
        if image_path:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
                # Detect mime type from extension
                ext = image_path.lower().split(".")[-1]
                mime_type = f"image/{ext if ext in ['jpeg', 'jpg', 'png', 'gif', 'webp'] else 'jpeg'}"
                image_content = f"data:{mime_type};base64,{image_data}"
        elif image_base64:
            # Assume it's already in data URI format or add it
            if image_base64.startswith("data:"):
                image_content = image_base64
            else:
                image_content = f"data:image/jpeg;base64,{image_base64}"
        elif image_url:
            image_content = image_url
        else:
            raise ValueError("Must provide image_path, image_base64, or image_url")

        # Build messages with image
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_content}}
            ]
        })

        result = self.chat(messages, **kwargs)
        return {
            "text": result["text"],
            "reasoning": result.get("reasoning"),
            "answer": result["text"]  # Alias for compatibility
        }

    @staticmethod
    def _extract_json_from_text(text: str) -> Dict[str, Any]:
        """Extract JSON from text, handling markdown code blocks"""
        import re

        # Try to find JSON in code blocks
        json_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(json_pattern, text, re.DOTALL)

        if match:
            json_str = match.group(1)
        else:
            # Try to find JSON object directly
            json_pattern = r"\{.*?\}"
            match = re.search(json_pattern, text, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                raise ValueError(f"Could not extract JSON from response: {text[:200]}")

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"JSON string: {json_str}")
            raise
