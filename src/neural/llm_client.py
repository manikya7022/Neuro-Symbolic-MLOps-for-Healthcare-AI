"""Async LLM client with Ollama integration for healthcare AI.

Provides reliable, production-ready LLM inference with:
- Async/await support for high throughput
- Automatic retries with exponential backoff
- Streaming response support
- Token-level probability extraction for uncertainty quantification
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class TokenInfo:
    """Information about a generated token."""
    
    token: str
    log_prob: float | None = None
    top_alternatives: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class LLMResponse:
    """Response from LLM inference."""
    
    text: str
    tokens: list[TokenInfo] = field(default_factory=list)
    model: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    @property
    def avg_log_prob(self) -> float | None:
        """Calculate average log probability across tokens."""
        probs = [t.log_prob for t in self.tokens if t.log_prob is not None]
        return sum(probs) / len(probs) if probs else None
    
    @property
    def min_log_prob(self) -> float | None:
        """Get minimum log probability (most uncertain token)."""
        probs = [t.log_prob for t in self.tokens if t.log_prob is not None]
        return min(probs) if probs else None


class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass


class LLMConnectionError(LLMClientError):
    """Connection error to LLM service."""
    pass


class LLMTimeoutError(LLMClientError):
    """Timeout waiting for LLM response."""
    pass


class LLMClient:
    """Async client for Ollama LLM inference.
    
    Features:
    - Automatic retry with exponential backoff
    - Fallback model support
    - Streaming responses
    - Token probability extraction
    - Response caching (optional)
    
    Example:
        ```python
        client = LLMClient()
        response = await client.generate("What is aspirin used for?")
        print(response.text)
        print(f"Confidence: {response.avg_log_prob}")
        ```
    """
    
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        fallback_model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        """Initialize LLM client.
        
        Args:
            host: Ollama server URL (default from config)
            model: Primary model name (default from config)
            fallback_model: Fallback model if primary fails
            timeout: Request timeout in seconds
        """
        config = settings.neural.ollama
        
        self.host = host or config.host
        self.model = model or config.model
        self.fallback_model = fallback_model or config.fallback_model
        self.timeout = timeout or config.timeout
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.max_retries = config.max_retries
        
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client (lazy initialization)."""
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(
                        base_url=self.host,
                        timeout=httpx.Timeout(self.timeout),
                    )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self) -> LLMClient:
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> list[str]:
        """List available models on Ollama server."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning("failed_to_list_models", error=str(e))
            return []
    
    @retry(
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        include_logprobs: bool = True,
    ) -> LLMResponse:
        """Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt for context
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens to generate
            model: Model to use (overrides default)
            include_logprobs: Whether to request log probabilities
        
        Returns:
            LLMResponse with generated text and metadata
        
        Raises:
            LLMConnectionError: If connection to Ollama fails
            LLMTimeoutError: If request times out
        """
        start_time = time.perf_counter()
        
        client = await self._get_client()
        model_to_use = model or self.model
        
        # Build request payload
        payload = {
            "model": model_to_use,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            },
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        # Request log probabilities for uncertainty quantification
        if include_logprobs:
            payload["options"]["logprobs"] = True
        
        try:
            response = await client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            
        except httpx.ConnectError as e:
            # Try fallback model
            if model_to_use != self.fallback_model:
                logger.warning(
                    "primary_model_failed_using_fallback",
                    primary=model_to_use,
                    fallback=self.fallback_model,
                    error=str(e),
                )
                return await self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=self.fallback_model,
                    include_logprobs=include_logprobs,
                )
            raise LLMConnectionError(f"Failed to connect to Ollama: {e}") from e
        
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"Request to Ollama timed out: {e}") from e
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Parse response
        text = data.get("response", "")
        
        # Extract token information if available
        tokens: list[TokenInfo] = []
        if "tokens" in data:
            for i, token in enumerate(data["tokens"]):
                log_prob = None
                if "logprobs" in data and i < len(data["logprobs"]):
                    log_prob = data["logprobs"][i]
                tokens.append(TokenInfo(token=token, log_prob=log_prob))
        
        # Get token counts
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        
        logger.debug(
            "llm_generation_complete",
            model=model_to_use,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        
        return LLMResponse(
            text=text,
            tokens=tokens,
            model=model_to_use,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the LLM.
        
        Yields tokens as they are generated for real-time UI updates.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: Model to use
        
        Yields:
            Generated tokens as strings
        """
        client = await self._get_client()
        model_to_use = model or self.model
        
        payload = {
            "model": model_to_use,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            },
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            async with client.stream("POST", "/api/generate", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
        
        except httpx.ConnectError as e:
            raise LLMConnectionError(f"Failed to connect to Ollama: {e}") from e
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"Streaming request timed out: {e}") from e
    
    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Chat completion with message history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            LLMResponse with assistant's reply
        """
        start_time = time.perf_counter()
        
        client = await self._get_client()
        model_to_use = model or self.model
        
        payload = {
            "model": model_to_use,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            },
        }
        
        try:
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        
        except httpx.ConnectError as e:
            raise LLMConnectionError(f"Failed to connect to Ollama: {e}") from e
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"Request timed out: {e}") from e
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        text = data.get("message", {}).get("content", "")
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        
        return LLMResponse(
            text=text,
            model=model_to_use,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


# Healthcare-specific system prompts
HEALTHCARE_SYSTEM_PROMPT = """You are a healthcare AI assistant specialized in medication safety and clinical decision support.

IMPORTANT GUIDELINES:
1. Always prioritize patient safety in your responses
2. Clearly identify potential drug interactions and contraindications
3. Note any allergies or conditions that may affect medication recommendations
4. Include appropriate warnings for high-risk medications
5. Recommend consulting a healthcare provider for serious concerns
6. Be precise with dosage information and never guess
7. Cite clinical evidence when available

You work alongside a symbolic verification system that checks your responses for safety compliance.
If you are uncertain about any medical information, clearly state your uncertainty."""


async def create_healthcare_client() -> LLMClient:
    """Create an LLM client configured for healthcare use cases."""
    client = LLMClient()
    
    # Check availability and pull model if needed
    if not await client.is_available():
        logger.warning("ollama_not_available")
    else:
        models = await client.list_models()
        if client.model not in models:
            logger.info("model_not_found_will_need_pull", model=client.model)
    
    return client
