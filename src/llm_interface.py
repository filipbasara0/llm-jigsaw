"""LLM interface for making puzzle-solving requests."""

import base64
import json
import logging
import re
from io import BytesIO
from typing import Optional, Any
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def request(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[np.ndarray],
    ) -> tuple[str, dict]:
        """
        Make a request to the LLM.
        
        Args:
            system_prompt: System-level instructions
            user_prompt: User message
            images: List of images to include
            
        Returns:
            Tuple of (response_text, usage_stats)
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (GPT-4o, etc.)."""
    
    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._client
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image to base64."""
        pil_image = Image.fromarray(image)
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def request(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[np.ndarray],
    ) -> tuple[str, dict]:
        client = self._get_client()
        
        # Build content with images
        content = []
        for img in images:
            base64_img = self._encode_image(img)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_img}",
                    "detail": "high"
                }
            })
        content.append({"type": "text", "text": user_prompt})
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            # max_tokens=4096,
            # temperature=0.1,
        )
        
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }
        
        return response.choices[0].message.content, usage


class AnthropicProvider(LLMProvider):
    """Anthropic API provider (Claude, etc.)."""
    
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client
    
    def _encode_image(self, image: np.ndarray) -> tuple[str, str]:
        """Encode image to base64 and return with media type."""
        pil_image = Image.fromarray(image)
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8"), "image/png"
    
    def request(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[np.ndarray],
    ) -> tuple[str, dict]:
        client = self._get_client()
        
        # Build content with images
        content = []
        for img in images:
            base64_img, media_type = self._encode_image(img)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_img,
                }
            })
        content.append({"type": "text", "text": user_prompt})
        
        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": content}]
        )
        
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }
        
        return response.content[0].text, usage


class GoogleProvider(LLMProvider):
    """Google Gemini API provider."""
    
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            from google import genai
            # genai.configure(api_key=self.api_key)
            self._client = genai.Client()
        return self._client
    
    def request(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[np.ndarray],
    ) -> tuple[str, dict]:
        client = self._get_client()
        
        # Build content
        content = []
        for img in images:
            pil_image = Image.fromarray(img)
            content.append(pil_image)
        content.append(f"{system_prompt}\n\n{user_prompt}")
        
        response = client.models.generate_content(model=self.model,contents=content)
        
        # Gemini doesn't always provide detailed usage
        usage = {
            "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0) if hasattr(response, "usage_metadata") else 0,
            "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0) if hasattr(response, "usage_metadata") else 0,
            "total_tokens": getattr(response.usage_metadata, "total_token_count", 0) if hasattr(response, "usage_metadata") else 0,
        }
        
        return response.text, usage


def get_provider(provider: str, model: str, api_key: str, base_url: Optional[str] = None) -> LLMProvider:
    """
    Factory function to get the appropriate provider.
    
    Args:
        provider: Provider name ('openai', 'anthropic', 'google')
        model: Model name
        api_key: API key
        base_url: Optional base URL override (for OpenAI-compatible APIs)
    """
    provider = provider.lower()
    
    if provider == "openai":
        return OpenAIProvider(model, api_key, base_url)
    elif provider == "anthropic":
        return AnthropicProvider(model, api_key)
    elif provider == "google":
        return GoogleProvider(model, api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")


class LLMInterface:
    """High-level interface for LLM puzzle-solving requests."""
    
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: Optional[str] = None
    ):
        """
        Initialize the LLM interface.
        
        Args:
            provider: LLM provider ('openai', 'anthropic', 'google')
            model: Model name
            api_key: API key
            base_url: Optional base URL for OpenAI-compatible APIs
        """
        self.provider_name = provider
        self.model = model
        self._provider = get_provider(provider, model, api_key, base_url)
        self._total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    
    def request_moves(
        self,
        image: np.ndarray,
        system_prompt: str,
        user_prompt: str,
        reference_image: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Request moves from the LLM.
        
        Args:
            image: Current puzzle state image
            system_prompt: System instructions
            user_prompt: User message with context
            reference_image: Optional original image for reference
            
        Returns:
            Parsed response dict with 'moves', 'reasoning', 'raw_response', 'usage'
        """
        images = [image]
        if reference_image is not None:
            images.append(reference_image)
        
        try:
            raw_response, usage = self._provider.request(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images=images
            )
            
            # Update total usage
            for key in self._total_usage:
                self._total_usage[key] += usage.get(key, 0)
            
            # Parse the response
            parsed = self._parse_response(raw_response)
            parsed["raw_response"] = raw_response
            parsed["usage"] = usage
            
            return parsed
            
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            return {
                "moves": [],
                "reasoning": None,
                "raw_response": str(e),
                "usage": {},
                "error": str(e)
            }
    
    def _parse_response(self, response: str) -> dict:
        """
        Parse the LLM response to extract moves.
        
        Args:
            response: Raw response string from LLM
            
        Returns:
            Dict with 'moves' list and 'reasoning' string
        """
        result = {
            "moves": [],
            "reasoning": None,
        }
        
        # Try to extract JSON from the response
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{[^{}]*"moves"[^{}]*\[.*?\][^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Last resort: find any JSON object
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    logger.warning("No JSON found in response")
                    return result
        
        try:
            data = json.loads(json_str)
            
            if "reasoning" in data:
                result["reasoning"] = data["reasoning"]
            
            if "moves" in data and isinstance(data["moves"], list):
                for move in data["moves"]:
                    if isinstance(move, dict) and move.get("op") == "swap":
                        a = move.get("a", "")
                        b = move.get("b", "")
                        if a and b:
                            result["moves"].append((a, b))
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
        
        return result
    
    def get_total_usage(self) -> dict:
        """Get cumulative token usage."""
        return self._total_usage.copy()
    
    def reset_usage(self) -> None:
        """Reset the usage counter."""
        self._total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
