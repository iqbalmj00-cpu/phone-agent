"""Phase-aware silence watchdog with a final activity check before disconnect."""
import asyncio


async def watch_silence(*, activity, busy, booked, speak, cancel, now, sleep=asyncio.sleep):
    observed = activity()
    phase = booked()
    anchor = now()
    nudged = False
    while True:
        await sleep(3)
        current, current_phase = activity(), booked()
        if current != observed or current_phase != phase:
            observed, phase, anchor, nudged = current, current_phase, now(), False
            continue
        if busy():
            # Agent speech pauses the timer but must not erase the nudge we sent.
            # Otherwise our own nudge starts a new nudge cycle forever.
            anchor = now()
            continue
        elapsed = now() - anchor
        nudge, goodbye = (15, 25) if phase else (30, 45)
        if elapsed >= goodbye:
            before = activity()
            await speak("The caller has been quiet. Say a brief neutral goodbye and invite them to call again. Do not claim an appointment or callback is confirmed.")
            await sleep(8)
            if activity() == before and not busy() and booked() == phase:
                await cancel()
                return
            observed, phase, anchor, nudged = activity(), booked(), now(), False
        elif elapsed >= nudge and not nudged:
            await speak("The caller has been quiet. Gently ask whether they are still there or need any more help.")
            nudged = True
