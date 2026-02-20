"""Sliding context window for long calls.

Keeps the last N turns in full and summarizes older turns into a compact recap
to keep token usage stable (~2K tokens) regardless of call length.
"""

from loguru import logger


MAX_FULL_TURNS = 10  # Keep this many recent turns verbatim
RECAP_REFRESH_INTERVAL = 5  # Re-summarize every N turns after threshold


def should_compress(messages: list[dict]) -> bool:
    """Check if context needs compression.

    Only considers user/assistant messages (not system).
    """
    conversation_turns = [m for m in messages if m["role"] in ("user", "assistant")]
    return len(conversation_turns) > MAX_FULL_TURNS * 2  # pairs of turns


async def compress_context(messages: list[dict], llm_client) -> list[dict]:
    """Compress older turns into a recap, keeping recent turns in full.

    Args:
        messages: Full message list including system prompt
        llm_client: OpenAI async client for summarization

    Returns:
        Compressed message list: [system, recap, recent turns]
    """
    system_messages = [m for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] in ("user", "assistant")]

    if len(conversation) <= MAX_FULL_TURNS * 2:
        return messages  # No compression needed

    # Split: older turns to summarize, recent turns to keep
    cutoff = len(conversation) - (MAX_FULL_TURNS * 2)
    older_turns = conversation[:cutoff]
    recent_turns = conversation[cutoff:]

    # Summarize older turns
    recap_prompt = (
        "Summarize this phone conversation so far in 2-3 sentences. "
        "Preserve all key facts: caller's name, phone number, address, "
        "appointment details, and any important requests. "
        "Be concise but don't lose critical booking information.\n\n"
    )
    for turn in older_turns:
        recap_prompt += f"{turn['role'].upper()}: {turn['content']}\n"

    try:
        response = await llm_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": recap_prompt}],
            max_tokens=200,
        )
        recap = response.choices[0].message.content
    except Exception as e:
        logger.error(f"Context compression failed: {e}")
        # Fallback: just keep recent turns without recap
        return system_messages + recent_turns

    # Build compressed context
    compressed = system_messages + [
        {"role": "system", "content": f"[Conversation recap] {recap}"},
        *recent_turns,
    ]

    logger.info(
        f"Context compressed: {len(conversation)} turns → "
        f"recap + {len(recent_turns)} recent turns"
    )
    return compressed
