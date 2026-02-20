"""Prosody-optimized Cartesia TTS service and text processor.

Subclasses CartesiaTTSService to inject `max_buffer_delay_ms` into
WebSocket messages for better inter-sentence transitions. Also provides
a minimal text processor for phone number formatting.

Pipeline:
  llm → SentenceAggregator → ProsodyProcessor → CartesiaContinuationTTS → transport
"""

import json
import re

from loguru import logger
from pipecat.frames.frames import Frame, TextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService


class CartesiaContinuationTTS(CartesiaTTSService):
    """CartesiaTTSService with max_buffer_delay_ms for smoother continuations.

    Overrides _build_msg to inject max_buffer_delay_ms into every
    WebSocket message sent to Cartesia. This controls how long Cartesia
    waits for more text before starting to generate audio, resulting in
    smoother transitions between sentences.
    """

    def __init__(self, *args, max_buffer_delay_ms: int = 150, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_buffer_delay_ms = max_buffer_delay_ms

    def _build_msg(
        self, text: str = "", continue_transcript: bool = True, add_timestamps: bool = True
    ):
        # Get the base JSON string from parent
        msg_str = super()._build_msg(text, continue_transcript, add_timestamps)
        # Parse, inject max_buffer_delay_ms, re-serialize
        msg = json.loads(msg_str)
        msg["max_buffer_delay_ms"] = self._max_buffer_delay_ms
        return json.dumps(msg)


# ── Phone number patterns ───────────────────────────────
_PHONE_PAREN = re.compile(r"\((\d{3})\)\s*(\d{3}[-.]?\d{4})")
_PHONE_DASH = re.compile(r"(\d{3})[-.](\.{3}[-.]?\d{4})")


def inject_prosody(text: str) -> str:
    """Wrap phone numbers in <spell> tags for clear digit reading."""
    original = text

    if "<spell" in text:
        return text

    text = _PHONE_PAREN.sub(
        r'<spell>(\1)</spell> <spell>\2</spell>',
        text,
    )
    text = _PHONE_DASH.sub(
        r'<spell>\1</spell> <spell>\2</spell>',
        text,
    )

    if text != original:
        logger.debug(f"Prosody: {original!r} → {text!r}")

    return text


class ProsodyProcessor(FrameProcessor):
    """Minimal text processor — only handles phone number <spell> tags."""

    def __init__(self):
        super().__init__()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
            original_text = frame.text
            modified_text = inject_prosody(original_text)

            if modified_text != original_text:
                await self.push_frame(TextFrame(text=modified_text), direction)
            else:
                await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)
