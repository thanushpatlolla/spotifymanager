"""Sprint 5: Claude API wrapper.

Provides a configured Anthropic client for LLM-powered features
like cluster naming and discovery context generation.
"""


def get_claude_client():
    """Create an authenticated Anthropic client."""
    raise NotImplementedError("Sprint 5")


def prompt_claude(client, system: str, user: str, max_tokens: int = 1024) -> str:
    """Send a prompt to Claude and return the text response."""
    raise NotImplementedError("Sprint 5")
