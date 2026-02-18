"""Claude API wrapper.

Provides a configured Anthropic client for LLM-powered features
like cluster naming and discovery context generation.
"""

import os

import anthropic

from src.data.db import load_env


def get_claude_client() -> anthropic.Anthropic:
    """Create an authenticated Anthropic client."""
    load_env()
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def prompt_claude(
    client: anthropic.Anthropic,
    system: str,
    user: str,
    max_tokens: int = 1024,
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """Send a prompt to Claude and return the text response."""
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return message.content[0].text
