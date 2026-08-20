"""Groq connection pool for efficient ChatGroq instance management."""
import os
import logging
import hashlib
import json
from typing import Optional, Dict, Any, Union
from threading import Lock
from pydantic import BaseModel
from langchain_groq import ChatGroq

from backend.model_profiles import apply_profile

logger = logging.getLogger(__name__)

# Environment variable configuration.
#
# This module holds the ONLY model/temperature literals in the codebase. Every
# caller passes None to inherit them, so `GROQ_DEFAULT_*` is the single place a
# default changes — see the resolution order in .env.example. Callers used to
# carry their own copies of the literal, which made these three unreachable:
# `model or GROQ_DEFAULT_MODEL` below can only fire if the caller's value is
# genuinely absent, and a caller with its own os.getenv fallback never is.
#
# llama-3.3-70b-versatile and llama-3.1-8b-instant were shut down by Groq on
# 2026-08-16. Do not reintroduce either as a default.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_DEFAULT_MODEL = os.getenv("GROQ_DEFAULT_MODEL", "openai/gpt-oss-120b")
GROQ_DEFAULT_TEMPERATURE = float(os.getenv("GROQ_DEFAULT_TEMPERATURE", "0.0"))
GROQ_DEFAULT_MAX_TOKENS = int(os.getenv("GROQ_DEFAULT_MAX_TOKENS", "4096"))


class GroqConnectionPool:
    """
    Connection pool for ChatGroq instances.

    Maintains reusable ChatGroq instances keyed by configuration
    to avoid instantiating new connections for every request.

    Supports JSON schema format for structured outputs.
    """

    def __init__(self):
        """Initialize the connection pool."""
        self._pool: Dict[str, ChatGroq] = {}
        self._lock = Lock()
        self._api_key = GROQ_API_KEY

        if not self._api_key:
            logger.warning("GROQ_API_KEY not found in environment variables")

    def _get_pool_key(
        self,
        model: str,
        temperature: float,
        json_schema: Optional[Dict] = None,
        **kwargs
    ) -> str:
        """
        Generate a unique key for the pool based on configuration.

        Args:
            model: Model name
            temperature: Model temperature
            json_schema: Optional JSON schema for structured output
            **kwargs: Additional configuration parameters

        Returns:
            Unique string key for this configuration
        """
        key_parts = [model, str(temperature)]

        # Hash the JSON schema if provided (schemas can be large)
        if json_schema:
            schema_str = json.dumps(json_schema, sort_keys=True)
            schema_hash = hashlib.md5(schema_str.encode()).hexdigest()[:8]
            key_parts.append(f"schema_{schema_hash}")

        # Add other kwargs
        for k, v in sorted(kwargs.items()):
            # Exclude sensitive/handled params AND callbacks: the Langfuse
            # handler is an identity-bearing object attached below, so keying
            # on it would fragment the pool (a new client per handler identity).
            if k not in ('api_key', 'format', 'callbacks'):
                key_parts.append(f"{k}={v}")

        return "_".join(key_parts)

    def _prepare_json_mode(
        self,
        format_param: Optional[Union[Dict, str]] = None
    ) -> Dict[str, Any]:
        """
        Prepare JSON mode configuration for ChatGroq.

        Args:
            format_param: Either 'json', a Pydantic schema dict, or None

        Returns:
            Dict with model_kwargs for JSON mode if needed
        """
        if format_param is None:
            return {}

        # If it's a string 'json', enable basic JSON mode
        if format_param == 'json':
            return {"model_kwargs": {"response_format": {"type": "json_object"}}}

        # If it's a dict (Pydantic schema), enable JSON mode
        # Note: Groq doesn't support full JSON schema validation like OpenAI,
        # but we enable JSON mode and the schema is used for prompt guidance
        if isinstance(format_param, dict):
            return {"model_kwargs": {"response_format": {"type": "json_object"}}}

        return {}

    def get_client(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        format: Optional[Union[Dict, str]] = None,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatGroq:
        """
        Get or create a ChatGroq client from the pool.

        Args:
            model: Groq model name (default from env: GROQ_DEFAULT_MODEL)
            temperature: Model temperature (default from env: GROQ_DEFAULT_TEMPERATURE)
            format: JSON format specification - either 'json' or a Pydantic schema dict
            api_key: Optional API key override
            max_tokens: Maximum tokens for response
            **kwargs: Additional ChatGroq configuration parameters

        Returns:
            ChatGroq instance from the pool
        """
        # Use defaults from environment if not provided
        model = model or GROQ_DEFAULT_MODEL
        temperature = temperature if temperature is not None else GROQ_DEFAULT_TEMPERATURE
        max_tokens = max_tokens or GROQ_DEFAULT_MAX_TOKENS

        # Reconcile the request with what this model family actually supports:
        # inject reasoning_format='hidden' for reasoning families (otherwise
        # deliberation lands in `content` and every json.loads in agents.py
        # fails into a silent fallback), and raise max_tokens to the family
        # floor. Applied BEFORE the pool key so two configs that differ only
        # after profiling never collide on one cached client.
        kwargs = apply_profile(model, {**kwargs, "max_tokens": max_tokens})
        max_tokens = kwargs.pop("max_tokens")

        # Use provided API key or fall back to environment variable
        used_api_key = api_key or self._api_key

        if not used_api_key:
            raise ValueError("No Groq API key provided. Set GROQ_API_KEY environment variable.")

        # Prepare JSON mode configuration
        json_config = self._prepare_json_mode(format)

        # Generate pool key
        pool_key = self._get_pool_key(
            model=model,
            temperature=temperature,
            json_schema=format if isinstance(format, dict) else None,
            max_tokens=max_tokens,
            **kwargs
        )

        # Thread-safe check and creation
        with self._lock:
            if pool_key not in self._pool:
                logger.info(f"Creating new ChatGroq instance: {pool_key}")

                # Merge kwargs with json config
                final_kwargs = {**kwargs, **json_config}

                # Langfuse tracing — attach the shared handler once, here at the
                # pool, so every call is traced automatically without callbacks
                # at each call site. No-op (None) unless keys are configured.
                # Excluded from the pool cache key above.
                from backend.observability import get_callback_handler
                handler = get_callback_handler()
                if handler is not None:
                    callbacks = list(final_kwargs.get("callbacks") or [])
                    callbacks.append(handler)
                    final_kwargs["callbacks"] = callbacks

                self._pool[pool_key] = ChatGroq(
                    model=model,
                    temperature=temperature,
                    api_key=used_api_key,
                    max_tokens=max_tokens,
                    **final_kwargs
                )
            else:
                logger.debug(f"Reusing existing ChatGroq instance: {pool_key}")

        return self._pool[pool_key]

    def clear_pool(self):
        """Clear all cached connections."""
        with self._lock:
            self._pool.clear()
            logger.info("Cleared all ChatGroq connections from pool")

    def get_pool_size(self) -> int:
        """Get the current number of connections in the pool."""
        with self._lock:
            return len(self._pool)

    def get_pool_keys(self) -> list:
        """Get all pool keys (for debugging)."""
        with self._lock:
            return list(self._pool.keys())


# Global connection pool instance
GROQ_CHAT = GroqConnectionPool()