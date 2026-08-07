"""A caller must not reach a booking that isn't theirs.

Before these gates existed, `lookup_appointment` keyed off the phone number the
caller *said*, not the number they were calling from. Stating any number
returned that customer's name, service address, appointment date and price — and
`resolve_job_id_from_lookup` then accepted any job id on trust, so the same
caller could reschedule or cancel a stranger's job.

The rule now: a spoken number is a search key, never evidence of identity.
"""
import unittest
from unittest.mock import AsyncMock, patch

from agent import handlers


OWNER = "+15551110000"
STRANGER = "+15559998888"

JOB = {
    "jobId": "job-1",
    "title": "Junk removal",
    "address": "12 Oak St, Houston, TX 77008",
    "scheduledDate": "2026-08-11",
    "timeSlot": "13:00-16:00",
    "status": "scheduled",
}
OTHER_JOB = {
    "jobId": "job-2",
    "title": "Junk removal",
    "address": "500 Pine Avenue, Houston, TX 77009",
    "scheduledDate": "2026-08-12",
    "timeSlot": "08:00-11:00",
    "status": "scheduled",
}


class CallerIdentityTests(unittest.TestCase):
    def setUp(self):
        handlers._call_contexts.clear()
        self.call_sid = "CA-IDENTITY"

    def tearDown(self):
        handlers._call_contexts.clear()

    def _call_from(self, number: str):
        handlers.set_call_context(self.call_sid, number, {"companyName": "Test Co"})

    # ── Who is calling ────────────────────────────────────────────

    def test_caller_id_matches_ignores_formatting(self):
        self._call_from(OWNER)
        for spoken in ("+1 555 111 0000", "(555) 111-0000", "555-111-0000", "5551110000"):
            with self.subTest(spoken=spoken):
                self.assertTrue(handlers.caller_id_matches(self.call_sid, spoken))

    def test_a_different_number_does_not_match(self):
        self._call_from(OWNER)
        self.assertFalse(handlers.caller_id_matches(self.call_sid, STRANGER))

    def test_calling_from_the_number_on_file_needs_no_challenge(self):
        self._call_from(OWNER)
        self.assertTrue(handlers.is_identity_verified(self.call_sid, OWNER))

    def test_a_stranger_is_not_verified_by_default(self):
        self._call_from(STRANGER)
        self.assertFalse(handlers.is_identity_verified(self.call_sid, OWNER))

    # ── The address challenge ─────────────────────────────────────

    def test_address_match_is_forgiving_about_how_people_speak(self):
        on_file = "12 Oak St, Houston, TX 77008"
        for spoken in ("12 Oak Street", "12 oak st", "12 Oak Street Houston", "12 Oak St."):
            with self.subTest(spoken=spoken):
                self.assertTrue(handlers.address_matches(spoken, on_file))

    def test_address_match_rejects_a_wrong_house_number(self):
        self.assertFalse(handlers.address_matches("14 Oak Street", "12 Oak St, Houston, TX 77008"))

    def test_address_match_rejects_a_wrong_street(self):
        self.assertFalse(handlers.address_matches("12 Pine Avenue", "12 Oak St, Houston, TX 77008"))

    def test_address_match_rejects_a_bare_house_number(self):
        # Guessing "12" must not pass just because the number is right.
        self.assertFalse(handlers.address_matches("12", "12 Oak St, Houston, TX 77008"))

    def test_address_match_rejects_empty_input(self):
        self.assertFalse(handlers.address_matches("", "12 Oak St, Houston, TX 77008"))
        self.assertFalse(handlers.address_matches("12 Oak Street", ""))

    # ── Mutations are gated on identity ───────────────────────────

    def test_stranger_cannot_reschedule_even_with_a_real_job_id(self):
        """The core regression: a known job id used to be accepted on trust."""
        self._call_from(STRANGER)
        handlers.remember_lookup_results(self.call_sid, OWNER, [JOB])

        job_id, error = handlers.resolve_job_id_from_lookup(self.call_sid, OWNER, "job-1")

        self.assertIsNone(job_id)
        self.assertEqual(error["error"], "identity_not_verified")
        self.assertTrue(error.get("fallback"))

    def test_owner_can_act_on_their_own_job(self):
        self._call_from(OWNER)
        handlers.remember_lookup_results(self.call_sid, OWNER, [JOB])

        job_id, error = handlers.resolve_job_id_from_lookup(self.call_sid, OWNER, "job-1")

        self.assertIsNone(error)
        self.assertEqual(job_id, "job-1")

    def test_a_job_id_never_returned_by_this_call_is_refused(self):
        """Stops a job id leaking in from anywhere other than this call's lookup."""
        self._call_from(OWNER)
        handlers.remember_lookup_results(self.call_sid, OWNER, [JOB])

        job_id, error = handlers.resolve_job_id_from_lookup(self.call_sid, OWNER, "job-somebody-else")

        self.assertIsNone(job_id)
        self.assertEqual(error["error"], "job_id_not_from_lookup")

    def test_passing_the_challenge_unlocks_the_booking(self):
        self._call_from(STRANGER)
        handlers.mark_identity_verified(self.call_sid, OWNER)
        handlers.remember_lookup_results(self.call_sid, OWNER, [JOB])

        job_id, error = handlers.resolve_job_id_from_lookup(self.call_sid, OWNER, "job-1")

        self.assertIsNone(error)
        self.assertEqual(job_id, "job-1")

    def test_verification_does_not_leak_across_calls(self):
        self._call_from(STRANGER)
        handlers.mark_identity_verified(self.call_sid, OWNER)

        other_sid = "CA-OTHER"
        handlers.set_call_context(other_sid, STRANGER, {"companyName": "Test Co"})
        self.assertFalse(handlers.is_identity_verified(other_sid, OWNER))

    # ── The withheld payload ──────────────────────────────────────

    def test_pending_verification_holds_details_out_of_the_model_s_reach(self):
        self._call_from(STRANGER)
        handlers.stash_pending_verification(
            self.call_sid, OWNER, {"customer": {"name": "Jane"}, "jobs": [JOB, OTHER_JOB]}
        )

        pending = handlers.get_pending_verification(self.call_sid)

        self.assertEqual(pending["phone"], OWNER)
        self.assertEqual(len(pending["jobs"]), 2)
        self.assertEqual(pending["customer"]["name"], "Jane")

    # ── The address challenge must not be guessable ────────────────

    def test_the_city_and_state_are_not_the_secret(self):
        """An operator serves one metro, so the city is public and constant.

        Matching on it reduced the challenge to "know the house number", which
        is visible from the street.
        """
        on_file = "12 Oak St, Houston, TX 77008"
        for spoken in ("12 Houston", "12 Texas", "12 TX 77008", "12 77008"):
            with self.subTest(spoken=spoken):
                self.assertFalse(handlers.address_matches(spoken, on_file))

    def test_a_generic_street_type_is_not_the_secret(self):
        """"500 Avenue" must not stand in for "500 Pine Avenue"."""
        cases = [
            ("500 Avenue", "500 Pine Avenue, Houston, TX 77009"),
            ("742 Terrace", "742 Evergreen Terrace, Springfield, IL 62704"),
            ("88 Drive", "88 Maple Drive, Austin, TX 78701"),
        ]
        for spoken, on_file in cases:
            with self.subTest(spoken=spoken):
                self.assertFalse(handlers.address_matches(spoken, on_file))

    def test_real_customers_still_get_through(self):
        """The tightening must not lock out people who genuinely know the address."""
        on_file = "12 Oak St, Houston, TX 77008"
        for spoken in ("12 Oak Street", "12 oak st", "12 Oak Street Houston", "12 Oak St."):
            with self.subTest(spoken=spoken):
                self.assertTrue(handlers.address_matches(spoken, on_file))

    # ── Scope: proving one booking proves only that booking ───────

    def test_verification_is_scoped_to_the_job_they_proved(self):
        handlers.set_call_context(self.call_sid, STRANGER, {"companyName": "Test Co"})
        handlers.mark_identity_verified(self.call_sid, OWNER, job_ids=["job-1"])

        self.assertEqual(handlers.verified_job_scope(self.call_sid, OWNER), {"job-1"})
        self.assertEqual(
            [j["jobId"] for j in handlers.limit_jobs_to_scope(self.call_sid, OWNER, [JOB, OTHER_JOB])],
            ["job-1"],
        )

    def test_calling_from_the_number_on_file_is_unscoped(self):
        """They are holding the phone the booking was made from — all of it is theirs."""
        self._call_from(OWNER)
        handlers.mark_identity_verified(self.call_sid, OWNER)

        self.assertIsNone(handlers.verified_job_scope(self.call_sid, OWNER))
        self.assertEqual(
            [j["jobId"] for j in handlers.limit_jobs_to_scope(self.call_sid, OWNER, [JOB, OTHER_JOB])],
            ["job-1", "job-2"],
        )


class FakeParams:
    def __init__(self, arguments):
        self.arguments = arguments
        self.results = []

    async def result_callback(self, payload):
        self.results.append(payload)


class LookupHandlerTests(unittest.IsolatedAsyncioTestCase):
    """End-to-end through the handlers, not just the helpers.

    A wiring mistake here is a disclosure, so the gate is exercised rather than
    read.
    """

    def setUp(self):
        handlers._call_contexts.clear()
        self.call_sid = "CA-LOOKUP"
        self.token = handlers._current_call_sid.set(self.call_sid)

    def tearDown(self):
        handlers._current_call_sid.reset(self.token)
        handlers._call_contexts.clear()

    def _call_from(self, number: str):
        handlers.set_call_context(
            self.call_sid, number, {"companyName": "Test Co", "agentSecret": "secret"}
        )

    def _dashboard_returns(self, payload):
        """Patch the one dashboard round-trip `handle_lookup_appointment` makes."""
        response = AsyncMock()
        response.status = 200
        session = AsyncMock()
        session.get = AsyncMock(return_value=response)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        return (
            patch.object(handlers.aiohttp, "ClientSession", return_value=session),
            patch.object(handlers, "_safe_json", AsyncMock(return_value=payload)),
        )

    async def _lookup(self, phone: str, payload: dict):
        params = FakeParams({"phone": phone})
        session_patch, json_patch = self._dashboard_returns(payload)
        with session_patch, json_patch:
            await handlers.handle_lookup_appointment(params)
        return params.results[-1]

    async def test_owner_gets_the_details_immediately(self):
        self._call_from(OWNER)
        result = await self._lookup(OWNER, {"found": True, "customer": {"name": "Jane"}, "jobs": [JOB]})

        self.assertTrue(result["found"])
        self.assertEqual(result["jobs"], [JOB])
        self.assertNotIn("verification_required", result)

    async def test_stranger_gets_no_details_at_all(self):
        """The heart of it: the model is never handed what it must not say."""
        self._call_from(STRANGER)
        result = await self._lookup(OWNER, {"found": True, "customer": {"name": "Jane"}, "jobs": [JOB]})

        self.assertTrue(result.get("verification_required"))
        self.assertNotIn("jobs", result)
        self.assertNotIn("customer", result)
        blob = str(result).lower()
        for secret in ("oak", "12 oak st", "jane", "2026-08-11", "13:00"):
            self.assertNotIn(secret.lower(), blob, f"leaked {secret!r} to the model")

    async def test_correct_address_unlocks_it(self):
        self._call_from(STRANGER)
        await self._lookup(OWNER, {"found": True, "customer": {"name": "Jane"}, "jobs": [JOB]})

        params = FakeParams({"address": "12 Oak Street"})
        await handlers.handle_verify_caller_identity(params)
        result = params.results[-1]

        self.assertTrue(result["verified"])
        self.assertEqual(result["jobs"], [JOB])
        self.assertTrue(handlers.is_identity_verified(self.call_sid, OWNER))

    async def test_wrong_address_reveals_nothing_and_hands_off(self):
        self._call_from(STRANGER)
        await self._lookup(OWNER, {"found": True, "customer": {"name": "Jane"}, "jobs": [JOB]})

        params = FakeParams({"address": "99 Elm Road"})
        await handlers.handle_verify_caller_identity(params)
        result = params.results[-1]

        self.assertFalse(result["verified"])
        self.assertEqual(result["error"], "identity_verification_failed")
        # Hand off to a person — explicitly not a callback, which would let an
        # unverified caller choose where the business rings back to.
        self.assertIn("transfer_to_human", result["message"])
        self.assertIn("do not offer a callback", result["message"].lower())
        self.assertNotIn("jobs", result)
        self.assertFalse(handlers.is_identity_verified(self.call_sid, OWNER))

    async def test_a_second_lookup_cannot_reopen_the_rest_of_the_account(self):
        """The account-takeover path, end to end.

        Passing the challenge narrowed the cache to the proved job — but the
        prompt tells the model to run `lookup_appointment` before every
        reschedule or cancel, and that second lookup used to refill the cache
        with every booking on the number. The caller then got the addresses and
        dates of bookings they had proved nothing about, and could cancel them.
        """
        self._call_from(STRANGER)
        both = {"found": True, "customer": {"name": "Jane"}, "jobs": [JOB, OTHER_JOB]}
        await self._lookup(OWNER, both)

        params = FakeParams({"address": "12 Oak Street"})
        await handlers.handle_verify_caller_identity(params)
        self.assertEqual([j["jobId"] for j in params.results[-1]["jobs"]], ["job-1"])

        # The step that used to undo it.
        again = await self._lookup(OWNER, both)
        self.assertEqual([j["jobId"] for j in again["jobs"]], ["job-1"])

        # Nothing about job-2 may reach the model, and it may not be acted on.
        job_id, error = handlers.resolve_job_id_from_lookup(self.call_sid, OWNER, None)
        self.assertEqual(job_id, "job-1")
        self.assertIsNone(error)

        refused_id, refused = handlers.resolve_job_id_from_lookup(self.call_sid, OWNER, "job-2")
        self.assertIsNone(refused_id)
        self.assertEqual(refused["error"], "job_id_not_from_lookup")

        blob = str(again) + str(error) + str(refused)
        for secret in ("pine", "500 Pine Avenue", "2026-08-12"):
            self.assertNotIn(secret.lower(), blob.lower(), f"leaked {secret!r}")

    async def test_a_failed_challenge_cannot_be_retried(self):
        """One attempt, then a person. Otherwise the address is brute-forceable."""
        self._call_from(STRANGER)
        await self._lookup(OWNER, {"found": True, "customer": {"name": "Jane"}, "jobs": [JOB]})

        first = FakeParams({"address": "99 Elm Road"})
        await handlers.handle_verify_caller_identity(first)
        self.assertEqual(first.results[-1]["error"], "identity_verification_failed")

        second = FakeParams({"address": "12 Oak Street"})  # now the RIGHT answer
        await handlers.handle_verify_caller_identity(second)
        self.assertFalse(second.results[-1]["verified"])
        self.assertEqual(second.results[-1]["error"], "no_pending_verification")

    async def test_a_fresh_lookup_cannot_re_arm_the_challenge(self):
        """Dropping the stash is not enough on its own.

        The prompt tells the model to call `lookup_appointment` before acting on
        a booking, so a caller who guessed wrong could simply have it looked up
        again and be handed another guess. Unlimited guesses against a house
        number and street name is not a challenge.
        """
        self._call_from(STRANGER)
        payload = {"found": True, "customer": {"name": "Jane"}, "jobs": [JOB]}
        await self._lookup(OWNER, payload)

        wrong = FakeParams({"address": "99 Elm Road"})
        await handlers.handle_verify_caller_identity(wrong)

        # The re-arm attempt.
        again = await self._lookup(OWNER, payload)
        self.assertEqual(again["error"], "identity_verification_failed")
        self.assertNotIn("jobs", again)
        self.assertNotIn("customer", again)
        self.assertIsNone(handlers.get_pending_verification(self.call_sid))

        retry = FakeParams({"address": "12 Oak Street"})  # the RIGHT answer
        await handlers.handle_verify_caller_identity(retry)
        self.assertFalse(retry.results[-1]["verified"])
        self.assertFalse(handlers.is_identity_verified(self.call_sid, OWNER))

    async def test_verifying_without_a_lookup_does_nothing(self):
        self._call_from(STRANGER)
        params = FakeParams({"address": "12 Oak Street"})
        await handlers.handle_verify_caller_identity(params)

        self.assertFalse(params.results[-1]["verified"])
        self.assertEqual(params.results[-1]["error"], "no_pending_verification")

    async def test_no_booking_found_says_so_without_a_challenge(self):
        self._call_from(STRANGER)
        result = await self._lookup(OWNER, {"found": False})

        self.assertFalse(result["found"])
        self.assertNotIn("verification_required", result)


if __name__ == "__main__":
    unittest.main()
