"""Prosody-optimized Cartesia TTS service and text processor.

Subclasses CartesiaTTSService to inject `max_buffer_delay_ms` into
WebSocket messages for better inter-sentence transitions. Also provides
text normalization for phone numbers and address readbacks.

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
        self,
        text: str = "",
        continue_transcript: bool = True,
        add_timestamps: bool = True,
        context_id: str = "",
    ):
        # Get the base JSON string from parent
        msg_str = super()._build_msg(
            text=text,
            continue_transcript=continue_transcript,
            add_timestamps=add_timestamps,
            context_id=context_id,
        )
        # Parse, inject max_buffer_delay_ms, re-serialize
        msg = json.loads(msg_str)
        msg["max_buffer_delay_ms"] = self._max_buffer_delay_ms
        return json.dumps(msg)


# ── Phone/address speech patterns ───────────────────────
_PHONE_NUMBER = re.compile(
    r"(?<!\d)(?:\+?1[\s.-]?)?\(?(\d{3})\)?[\s.-]*(\d{3})[\s.-]*(\d{4})(?!\d)"
)
_STREET_SUFFIX_PATTERN = (
    r"street|st\.?|avenue|ave\.?|road|rd\.?|drive|dr\.?|lane|ln\.?|"
    r"court|ct\.?|circle|cir\.?|boulevard|blvd\.?|way|place|pl\.?|"
    r"terrace|ter\.?|trail|trl\.?|parkway|pkwy\.?|highway|hwy\.?"
)
_ADDRESS_NUMBER = re.compile(
    rf"\b(\d{{1,6}})(?=\s+(?:[A-Za-z0-9'.-]+\s+){{0,6}}(?:{_STREET_SUFFIX_PATTERN})\b)",
    re.IGNORECASE,
)
_ADDRESS_CONTEXT = re.compile(rf"\b(?:{_STREET_SUFFIX_PATTERN})\b", re.IGNORECASE)
_ZIP_CODE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
_STATE_ABBR_TO_NAME = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
}
_STATE_ZIP_CODE = re.compile(
    r"\b("
    + "|".join([*_STATE_ABBR_TO_NAME.keys(), *_STATE_ABBR_TO_NAME.values()])
    + r")\s+(\d{5})(?:-\d{4})?\b",
    re.IGNORECASE,
)


def _spell_phone(match: re.Match) -> str:
    area, prefix, line = match.groups()
    return (
        f"<spell>{area}</spell> <break time=\"120ms\" /> "
        f"<spell>{prefix}</spell> <break time=\"120ms\" /> "
        f"<spell>{line}</spell>"
    )


def _spell_digits(digits: str) -> str:
    return f"<spell>{digits}</spell>"


def _space_digits(match: re.Match) -> str:
    return " ".join(match.group(1))


def _expand_state(match: re.Match) -> str:
    return _STATE_ABBR_TO_NAME.get(match.group(1).upper(), match.group(1))


def format_address_for_speech(address: str) -> str:
    """Return a caller-facing address readback without changing stored data."""
    text = (address or "").replace(", USA", "").replace(", US", "").strip()
    if not text:
        return ""

    text = _ADDRESS_NUMBER.sub(_space_digits, text)
    text = re.sub(r"\b([A-Z]{2})\b(?=\s+\d{5}(?:-\d{4})?\b)", _expand_state, text)
    text = _ZIP_CODE.sub(_space_digits, text)
    return text


def _spell_address_number(match: re.Match) -> str:
    return _spell_digits(match.group(1))


def _spell_state_zip(match: re.Match) -> str:
    return f"{match.group(1)} {_spell_digits(match.group(2))}"


def inject_prosody(text: str) -> str:
    """Wrap phone numbers and address digits for clear spoken readback."""
    original = text

    if "<spell" not in text:
        text = _PHONE_NUMBER.sub(_spell_phone, text)

    if _ADDRESS_CONTEXT.search(text):
        text = _ADDRESS_NUMBER.sub(_spell_address_number, text)
        text = re.sub(r"\b([A-Z]{2})\b(?=\s+\d{5}(?:-\d{4})?\b)", _expand_state, text)
        text = _STATE_ZIP_CODE.sub(_spell_state_zip, text)

    if text != original:
        logger.debug(f"Prosody: {original!r} → {text!r}")

    return text


class ProsodyProcessor(FrameProcessor):
    """Text processor for phone numbers and address readback digits."""

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
