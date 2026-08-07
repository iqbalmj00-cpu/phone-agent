"""Every call must be finalized, including the ones the agent ends itself.

The post-call work — summary, dashboard call log, releasing the per-call context —
used to live inside the `on_client_disconnected` handler alone. That handler only
fires when the caller hangs up. When the agent ends the call (duration cap, either
silence watcher, or a transfer) it calls `task.cancel()`, and the resulting
`CancelledError` is a `BaseException`, so it escapes the `except Exception` inside
Pipecat's websocket transport and the disconnect event never fires. Those calls were
left with no summary, no duration, no outcome — and their PipelineTask was never
released, because `clear_call_context` lived in the same skipped block.

`run_post_call` is now module-level and takes its state as arguments, so it can be
exercised here. `_OnceGuard` keeps the two entry paths from both running it.

Not covered here: that the `try/finally` around `runner.run(task)` actually fires on
those four cancel paths. That needs a live pipeline. It rests on Pipecat's
`PipelineTask.run` returning normally after an in-app cancel.
"""
import asyncio
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import bot
from agent import handlers


CALL_SID = "CA-FINALIZE"
CALLER = "+15551110000"
TWILIO_NUMBER = "+15552223333"
TZ = "America/Chicago"


class FakeContext:
    """Stands in for the Pipecat LLMContext — only `.messages` is read."""

    def __init__(self, messages=None):
        self.messages = messages if messages is not None else []


class OnceGuardTests(unittest.TestCase):
    def test_only_the_first_claim_wins(self):
        guard = bot._OnceGuard()
        self.assertTrue(guard.claim())
        self.assertFalse(guard.claim())
        self.assertFalse(guard.claim())

    def test_guards_are_independent(self):
        """One per call — a shared flag would silence the second caller's log."""
        first, second = bot._OnceGuard(), bot._OnceGuard()
        self.assertTrue(first.claim())
        self.assertTrue(second.claim())


class RunPostCallTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        handlers._call_contexts.clear()
        handlers._pending_transfers.clear()
        handlers.set_call_context(
            CALL_SID, CALLER, {"companyName": "Test Co", "twilioNumber": TWILIO_NUMBER}
        )

    def tearDown(self):
        handlers._call_contexts.clear()
        handlers._pending_transfers.clear()

    async def _run(self, *, messages=None, summary="Caller asked about pricing."):
        """Drive run_post_call with the two network calls stubbed out."""
        started = datetime.now(ZoneInfo(TZ)) - timedelta(seconds=90)
        logged = AsyncMock()
        with patch.object(bot, "_summarize_call_with_anthropic", AsyncMock(return_value=summary)), \
             patch.object(bot, "log_call_to_dashboard", logged), \
             patch.object(bot, "_delayed_followup_sms", AsyncMock()) as followup:
            await bot.run_post_call(
                call_id=CALL_SID,
                caller_number=CALLER,
                client_config={"companyName": "Test Co", "twilioNumber": TWILIO_NUMBER},
                context=FakeContext(messages if messages is not None else [
                    {"role": "user", "content": "How much for a couch?"},
                    {"role": "assistant", "content": "Around a hundred and fifty."},
                ]),
                call_start_time=started,
                timezone=TZ,
            )
            # Let any follow-up SMS task scheduled by create_task actually start.
            await asyncio.sleep(0)
        return logged, followup

    async def test_a_booked_call_is_logged_as_booked(self):
        handlers.mark_booking_complete(CALL_SID, {"caller_name": "Jane", "appointment_date": "2026-08-11"})

        logged, _ = await self._run()

        self.assertEqual(logged.await_count, 1)
        kwargs = logged.await_args.kwargs
        self.assertEqual(kwargs["outcome"], "booked")
        self.assertEqual(kwargs["twilio_call_sid"], CALL_SID)
        self.assertEqual(kwargs["from_number"], CALLER)
        self.assertEqual(kwargs["to_number"], TWILIO_NUMBER)
        self.assertEqual(kwargs["caller_name"], "Jane")
        self.assertEqual(kwargs["appointment_date"], "2026-08-11")
        self.assertGreater(kwargs["duration"], 0)

    async def test_an_ordinary_call_is_logged_as_info_only(self):
        logged, _ = await self._run()

        self.assertEqual(logged.await_args.kwargs["outcome"], "info_only")

    async def test_a_dashboard_owned_transfer_is_not_logged_here(self):
        """The dashboard finalizes its own transfers.

        A late write from this side downgrades dashboard_answered back to
        dialing, so the call log must be skipped — not merely duplicated.
        """
        handlers.mark_dashboard_transfer_ownership(CALL_SID, "ai_transfer", reason="caller_requested")

        logged, _ = await self._run()

        self.assertEqual(logged.await_count, 0)

    async def test_the_per_call_context_is_always_released(self):
        """This is the memory leak. The context holds the PipelineTask, and the
        PipelineTask holds a per-call voice-activity-detection session."""
        self.assertIn(CALL_SID, handlers._call_contexts)

        await self._run()

        self.assertNotIn(CALL_SID, handlers._call_contexts)

    async def test_the_context_is_released_on_a_transfer_too(self):
        handlers.mark_dashboard_transfer_ownership(CALL_SID, "ai_transfer", reason="caller_requested")

        await self._run()

        self.assertNotIn(CALL_SID, handlers._call_contexts)

    async def test_follow_up_sms_is_sent_when_they_consented_and_did_not_book(self):
        handlers.mark_sms_consent(CALL_SID, True)

        _, followup = await self._run()

        self.assertEqual(followup.call_count, 1)

    async def test_no_follow_up_sms_without_consent(self):
        _, followup = await self._run()

        self.assertEqual(followup.call_count, 0)

    async def test_no_follow_up_sms_after_a_booking(self):
        """They just booked — the follow-up is for callers who didn't."""
        handlers.mark_sms_consent(CALL_SID, True)
        handlers.mark_booking_complete(CALL_SID, {})

        _, followup = await self._run()

        self.assertEqual(followup.call_count, 0)

    async def test_a_booking_still_in_flight_is_waited_for(self):
        """The ghost-booking guard.

        A caller can hang up in the second between `create_booking` reaching the
        dashboard and its response coming back. Without the wait, the booking
        exists but the call log says `info_only`, and the operator has a job
        nobody knows about.
        """
        async def _slow_booking():
            await asyncio.sleep(0.02)
            handlers.mark_booking_complete(CALL_SID, {"caller_name": "Jane"})

        handlers.add_inflight_task(CALL_SID, asyncio.create_task(_slow_booking()))

        logged, _ = await self._run()

        self.assertEqual(logged.await_args.kwargs["outcome"], "booked")
        self.assertEqual(logged.await_args.kwargs["caller_name"], "Jane")

    async def test_a_hung_tool_call_does_not_block_the_log_forever(self):
        """The wait is bounded — a stuck request must not cost us the call log."""
        async def _never_returns():
            await asyncio.sleep(30)

        stuck = asyncio.create_task(_never_returns())
        handlers.add_inflight_task(CALL_SID, stuck)

        with patch.object(bot, "INFLIGHT_TASK_WAIT_SECONDS", 0.01):
            logged, _ = await self._run()

        self.assertEqual(logged.await_count, 1)
        stuck.cancel()

    async def test_a_silent_call_still_produces_a_log(self):
        logged, _ = await self._run(messages=[])

        self.assertEqual(logged.await_count, 1)
        self.assertIn("No caller conversation captured", logged.await_args.kwargs["summary"])

    async def test_a_slow_summary_falls_back_to_the_transcript(self):
        """An operator with a rough transcript is better off than one with nothing."""
        async def _never_finishes(_messages):
            await asyncio.sleep(5)
            return "unreachable"

        started = datetime.now(ZoneInfo(TZ)) - timedelta(seconds=90)
        logged = AsyncMock()
        with patch.object(bot, "SUMMARY_WAIT_SECONDS", 0.01), \
             patch.object(bot, "_summarize_call_with_anthropic", _never_finishes), \
             patch.object(bot, "log_call_to_dashboard", logged), \
             patch.object(bot, "_delayed_followup_sms", AsyncMock()):
            await bot.run_post_call(
                call_id=CALL_SID,
                caller_number=CALLER,
                client_config={"companyName": "Test Co", "twilioNumber": TWILIO_NUMBER},
                context=FakeContext([
                    {"role": "user", "content": "Do you take mattresses?"},
                ]),
                call_start_time=started,
                timezone=TZ,
            )

        summary = logged.await_args.kwargs["summary"]
        self.assertIn("AI summary unavailable", summary)
        self.assertIn("mattresses", summary)


if __name__ == "__main__":
    unittest.main()
