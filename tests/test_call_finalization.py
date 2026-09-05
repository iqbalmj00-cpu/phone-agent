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


class _PostCallHarness(unittest.IsolatedAsyncioTestCase):
    """Shared setup. No tests of its own, so nothing runs twice."""

    def setUp(self):
        handlers._call_contexts.clear()
        # A terminal transfer state outlives clear_call_context by design, so it
        # leaks between tests that reuse this call sid unless it is cleared too.
        handlers._terminal_transfer_states.clear()
        handlers.set_call_context(
            CALL_SID, CALLER, {"companyName": "Test Co", "twilioNumber": TWILIO_NUMBER}
        )

    def tearDown(self):
        handlers._call_contexts.clear()
        handlers._terminal_transfer_states.clear()

    async def _run(self, *, messages=None, summary="Caller asked about pricing."):
        """Drive run_post_call with the two network calls stubbed out."""
        started = datetime.now(ZoneInfo(TZ)) - timedelta(seconds=90)
        logged = AsyncMock()
        self.lead = AsyncMock(return_value=True)
        with patch.object(bot, "_summarize_call_with_anthropic", AsyncMock(return_value=summary)), \
             patch.object(bot, "create_lead_from_call", self.lead), \
             patch.object(bot, "log_call_to_dashboard", logged):
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
        return logged


class RunPostCallTests(_PostCallHarness):
    async def test_a_booked_call_is_logged_as_booked(self):
        handlers.mark_booking_complete(CALL_SID, {"caller_name": "Jane", "appointment_date": "2026-08-11"})

        logged = await self._run()

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
        logged = await self._run()

        self.assertEqual(logged.await_args.kwargs["outcome"], "info_only")

    async def test_a_dashboard_owned_transfer_is_not_logged_here(self):
        """The dashboard finalizes its own transfers.

        A late write from this side downgrades dashboard_answered back to
        dialing, so the call log must be skipped — not merely duplicated.
        """
        handlers.mark_dashboard_transfer_ownership(CALL_SID, "ai_transfer", reason="caller_requested")

        logged = await self._run()

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

        logged = await self._run()

        self.assertEqual(logged.await_args.kwargs["outcome"], "booked")
        self.assertEqual(logged.await_args.kwargs["caller_name"], "Jane")

    async def test_a_hung_tool_call_does_not_block_the_log_forever(self):
        """The wait is bounded — a stuck request must not cost us the call log."""
        async def _never_returns():
            await asyncio.sleep(30)

        stuck = asyncio.create_task(_never_returns())
        handlers.add_inflight_task(CALL_SID, stuck)

        with patch.object(bot, "INFLIGHT_TASK_WAIT_SECONDS", 0.01):
            logged = await self._run()

        self.assertEqual(logged.await_count, 1)
        stuck.cancel()

    async def test_a_silent_call_still_produces_a_log(self):
        logged = await self._run(messages=[])

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
             patch.object(bot, "log_call_to_dashboard", logged):
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


class CallLogRetryTests(_PostCallHarness):
    """The retry has to key on the result, not on a timeout that cannot fire.

    `log_call_to_dashboard` carries its own 10s client timeout and turns every
    failure into `False`, so the 12s `wait_for` around it almost never raises.
    An earlier version retried on `asyncio.TimeoutError` and was therefore dead
    code for the dashboard hang it existed to cover.
    """

    async def _run_with(self, results):
        logged = AsyncMock(side_effect=list(results))
        with patch.object(bot, "_summarize_call_with_anthropic", AsyncMock(return_value="x")), \
             patch.object(bot, "create_lead_from_call", AsyncMock(return_value=True)), \
             patch.object(bot, "log_call_to_dashboard", logged):
            await bot.run_post_call(
                call_id=CALL_SID,
                caller_number=CALLER,
                client_config={"companyName": "Test Co", "twilioNumber": TWILIO_NUMBER},
                context=FakeContext([{"role": "user", "content": "hi"}]),
                call_start_time=datetime.now(ZoneInfo(TZ)) - timedelta(seconds=90),
                timezone=TZ,
            )
        return logged

    async def test_a_refused_log_is_retried_once(self):
        logged = await self._run_with([False, True])
        self.assertEqual(logged.await_count, 2)

    async def test_a_successful_log_is_not_retried(self):
        logged = await self._run_with([True])
        self.assertEqual(logged.await_count, 1)

    async def test_it_gives_up_after_two_attempts(self):
        """Never a third: each POST is cheap, but a loop is not."""
        logged = await self._run_with([False, False])
        self.assertEqual(logged.await_count, 2)


class LeadCaptureGateTests(_PostCallHarness):
    """Who becomes a lead, and — more importantly — who does not.

    D12 supplies the name gate. D27 supplies the outcome gate: a caller already
    sitting in the dashboard's Callback Queue must not also be captured as a
    lead, or the follow-up cron cold-texts someone the operator is meant to be
    ringing back.
    """

    async def test_a_named_caller_who_did_not_book_becomes_a_lead(self):
        handlers.mark_caller_name(CALL_SID, "Jane")

        await self._run(summary="Caller wants a garage cleared next week.")

        self.lead.assert_awaited_once()
        kwargs = self.lead.await_args.kwargs
        self.assertEqual(kwargs["name"], "Jane")
        self.assertEqual(kwargs["phone"], CALLER)
        self.assertEqual(kwargs["description"], "Caller wants a garage cleared next week.")

    async def test_no_name_means_no_lead(self):
        """D12, enforced in code rather than trusted to the prompt."""
        await self._run()
        self.lead.assert_not_awaited()

    async def test_a_booked_caller_is_not_also_a_lead(self):
        handlers.mark_caller_name(CALL_SID, "Jane")
        handlers.mark_booking_complete(CALL_SID, {"caller_name": "Jane"})

        await self._run()

        self.lead.assert_not_awaited()

    async def test_an_after_hours_refusal_is_not_a_lead(self):
        """D27. This caller is already in the Callback Queue."""
        handlers.mark_caller_name(CALL_SID, "Jane")
        handlers.mark_transfer_state(
            CALL_SID, "suppressed_after_hours", "asked for a person", callback_requested=True
        )

        await self._run()

        self.lead.assert_not_awaited()

    async def test_a_failed_transfer_is_not_a_lead(self):
        handlers.mark_caller_name(CALL_SID, "Jane")
        handlers.mark_transfer_state(CALL_SID, "failed", "transfer blew up", callback_requested=True)

        await self._run()

        self.lead.assert_not_awaited()

    async def test_a_scheduled_callback_is_not_a_lead(self):
        handlers.mark_caller_name(CALL_SID, "Jane")
        handlers.mark_callback_requested(CALL_SID, "2026-09-10T14:00:00-05:00", "wants a call back")

        await self._run()

        self.lead.assert_not_awaited()

    async def test_a_failed_summary_still_produces_a_lead_without_a_description(self):
        """The description renders verbatim to the operator, so no machine text."""
        handlers.mark_caller_name(CALL_SID, "Jane")

        await self._run(summary="Summary unavailable")

        self.lead.assert_awaited_once()
        self.assertEqual(self.lead.await_args.kwargs["description"], "")

    async def test_the_name_is_read_before_the_context_is_cleared(self):
        """`clear_call_context` runs one statement later and wipes the name."""
        handlers.mark_caller_name(CALL_SID, "Jane")

        await self._run()

        self.assertEqual(self.lead.await_args.kwargs["name"], "Jane")
        self.assertEqual(handlers.get_caller_name(CALL_SID), "")


if __name__ == "__main__":
    unittest.main()
