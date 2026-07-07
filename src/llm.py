"""Groq-backed LLM client for answer generation."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from groq import Groq

from src.utils import get_logger

logger = get_logger(__name__)


class GroqLLM:
    """Wrapper around the Groq chat-completions API.

    The API key is read from the ``GROQ_API_KEY`` environment variable and is
    never hardcoded.
    """

    def __init__(self, model: str) -> None:
        """Initialize the Groq client.

        Args:
            model: The Groq chat model to use for completions.

        Raises:
            RuntimeError: If ``GROQ_API_KEY`` is not set in the environment.
        """
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        self.model = model
        self._client = Groq(api_key=api_key)

    def generate(self, prompt: str) -> str:
        """Generate an answer for the given prompt.

        Args:
            prompt: The fully-built prompt to send to the model.

        Returns:
            The model's response text, stripped of surrounding whitespace.
        """
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = completion.choices[0].message.content
        return content.strip() if content else "I don't know."
