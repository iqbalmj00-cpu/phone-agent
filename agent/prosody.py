"""Prosody-optimized Cartesia TTS service and text processor.

Subclasses CartesiaTTSService to inject `max_buffer_delay_ms` into
WebSocket messages for better inter-sentence transitions. Also provides
text normalization for phone numbers and address readbacks.

Pipeline:
  llm → SentenceAggregator → ProsodyProcessor → CartesiaContinuationTTS → transport
"""

import json
import re
from urllib.parse import urlsplit

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
# A street name can run long — "1234 North Martin Luther King Jr Boulevard" —
# so the gap between the number and its suffix stays wide. What stopped
# "$300 either way" being spelled as "three, zero, zero dollars" is the
# currency guard here plus the narrowed suffix list below, not a shorter window.
_ADDRESS_NUMBER = re.compile(
    rf"\b(?<![$])(\d{{1,6}})(?=\s+(?:[A-Za-z0-9'.-]+\s+){{0,6}}(?:{_STREET_SUFFIX_PATTERN})\b)",
    re.IGNORECASE,
)
# Only suffixes that are not ordinary English words may arm digit-spelling for a
# whole sentence. "way", "drive", "place", "court", "circle" and "trail" all
# appear in normal speech — "either way", "on the drive", "in the first place" —
# and each one used to turn every number in the sentence into spelled digits.
_UNAMBIGUOUS_STREET_SUFFIX = (
    r"street|st\.|avenue|ave\.?|road|rd\.?|lane|ln\.?|boulevard|blvd\.?|"
    r"terrace|ter\.?|parkway|pkwy\.?|highway|hwy\.?"
)
_ADDRESS_CONTEXT = re.compile(rf"\b(?:{_UNAMBIGUOUS_STREET_SUFFIX})\b", re.IGNORECASE)
_ZIP_CODE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
# Area code and exchange both start 2-9 under the NANP. Length alone would let
# a nine-digit number through as a plausible-looking ten-digit one.
_NANP_NUMBER = re.compile(r"[2-9]\d{2}[2-9]\d{2}\d{4}")
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


def format_website_for_speech(url: str) -> str:
    """Reduce a configured website URL to the form a person says out loud.

    "https://www.acmehauling.com/" becomes "acmehauling.com". The dashboard
    stores whatever the operator typed, so both bare and full forms arrive.

    Returns "" when the value cannot be spoken as a domain, so the caller can
    leave the whole sentence out rather than read a fragment aloud.
    """
    candidate = (url or "").strip()
    if not candidate:
        return ""

    # urlsplit only finds a host after "//", which a bare domain has no reason
    # to carry.
    parsed = urlsplit(candidate if "//" in candidate else f"//{candidate}")
    host = (parsed.netloc or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]

    if "." not in host or " " in host:
        logger.warning(
            f"websiteUrl is not a speakable domain, so the agent will not "
            f"mention a website: {url!r}"
        )
        return ""

    return f"{host}{(parsed.path or '').rstrip('/')}"


def format_phone_for_speech(number: str) -> str:
    """Reduce a stored phone number to the form the agent can safely say aloud.

    "+15125551234" becomes "(512) 555-1234" — the grouping a person writes and
    a model repeats verbatim, which is what lets `inject_prosody` find it and
    spell it in three chunks. Handing the model raw E.164 instead invites it to
    paraphrase the "+1" into words, and the spelling never fires.

    Returns "" for anything that is not a ten-digit North American number, so
    the caller hears nothing rather than a wrong number. Length alone is not
    the bar. "+442071234567" is too long and would be read out raw, but
    "+1512555123" is one digit short and strips to ten characters that look
    valid — it renders as (151) 255-5123, a different number said with total
    confidence. Requiring the area code and exchange to start 2-9, as NANP
    does, is what separates the two.
    """
    raw = (number or "").strip()
    if not raw:
        return ""

    digits = re.sub(r"\D", "", raw)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]

    if not _NANP_NUMBER.fullmatch(digits):
        logger.warning(
            f"twilioNumber is not a North American number the voice pipeline "
            f"can spell, so the agent will not say a phone number: {number!r}"
        )
        return ""

    return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"


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
