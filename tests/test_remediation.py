"""Controlled phone regressions. All network/provider boundaries are replaced by fixtures."""
import asyncio
import json
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import bot
from agent import handlers, dashboard_redirect
from agent.booking_outcome import booking_outcome, sms_guidance
from agent.prompt import build_system_prompt
from agent.silence import watch_silence


class Params:
    def __init__(self, **arguments):
        self.arguments, self.results = arguments, []
    async def result_callback(self, result):
        self.results.append(result)


class Response:
    def __init__(self, status, body):
        self.status, self.body = status, body
    async def text(self):
        return self.body if isinstance(self.body, str) else json.dumps(self.body)
    async def json(self, **kwargs):
        return json.loads(await self.text())


class Network:
    def __init__(self, responses):
        self.responses, self.requests = iter(responses), []
    async def __aenter__(self): return self
    async def __aexit__(self, *args): return False
    async def post(self, url, **kwargs):
        self.requests.append((url, kwargs.get('json')))
        response = next(self.responses)
        if isinstance(response, Exception): raise response
        return response
    async def get(self, url, **kwargs):
        self.requests.append((url, kwargs.get('params')))
        return next(self.responses)


class OutcomeTests(unittest.TestCase):
    def test_semantic_matrix(self):
        for status, data, outcome in [
            (201, {}, 'uncertain'), (201, {'success':False}, 'uncertain'),
            (201, {'success':True, 'scheduled':False, 'leadId':'L'}, 'pending'),
            (200, {'success':True, 'scheduled':True, 'jobId':'J'}, 'scheduled'),
            (201, {'success':True, 'scheduled':True}, 'uncertain'),
            (202, {'success':True, 'scheduled':True, 'jobId':'J'}, 'uncertain'),
            (200, {'success':False, 'feasibility':{}}, 'refused'),
            (409, {'success':False, 'outcome':'refused'}, 'refused'),
        ]:
            with self.subTest(status=status, data=data): self.assertEqual(booking_outcome(status,data),outcome)
    def test_no_sms_delivery_claim(self):
        self.assertIn('Do not promise',sms_guidance({}))
        self.assertIn('queued',sms_guidance({'sms':{'status':'queued'}}))
        self.assertIn('delivery is not confirmed',sms_guidance({'sms':{'status':'accepted'}}))
    def test_terminal_retention(self):
        handlers._terminal_transfer_states.clear()
        with patch.object(handlers.time,'monotonic',return_value=100), patch.object(handlers,'_TERMINAL_TRANSFER_MAX_ENTRIES',2):
            for sid in ('a','b','c'): handlers.remember_terminal_transfer_state(sid,'failed')
            self.assertEqual(set(handlers._terminal_transfer_states),{'b','c'})
            self.assertEqual(handlers.get_terminal_transfer_state('c')['transfer_status'],'failed')
        with patch.object(handlers.time,'monotonic',return_value=3701):
            self.assertEqual(handlers.get_terminal_transfer_state('c'),{})


class HandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        handlers._call_contexts.clear(); handlers._terminal_transfer_states.clear()
        self.sid='CA-fixture'; self.origin='+15551110000'
        self.config={'_clientId':'org-a','agentSecret':'fixture','companyName':'Fixture Co','timezone':'America/Chicago','businessDays':list(range(7)),'businessStart':0,'businessEnd':23,'dumpsterRentalsEnabled':True}
        handlers.set_call_context(self.sid,self.origin,self.config)
        self.token=handlers._current_call_sid.set(self.sid)
    def tearDown(self):
        handlers._current_call_sid.reset(self.token)
        handlers._call_contexts.clear(); handlers._terminal_transfer_states.clear()
    def booking(self, **extra):
        return Params(name='Alex',phone=self.origin,address='10 Oak St',date='2026-09-10',time='13:00-16:00',description='furniture',type='pickup',**extra)
    async def run_booking(self, params, network):
        with patch.object(handlers.aiohttp,'ClientSession',return_value=network):
            await handlers.handle_create_booking(params)
        return params.results[-1]
    async def test_pending_and_html_never_mark_booked(self):
        for body in ({'success':True,'scheduled':False,'leadId':'L'}, '<html>proxy</html>', {'success':False}):
            result=await self.run_booking(self.booking(), Network([Response(201,body)]))
            self.assertFalse(handlers.is_booking_complete(self.sid))
            self.assertNotEqual(result.get('scheduled'),True)
    async def test_response_loss_reuses_intent_and_blocks_changed_details(self):
        net=Network([TimeoutError(),Response(200,{'success':True,'scheduled':True,'jobId':'J'})])
        await self.run_booking(self.booking(),net)
        changed=self.booking(); changed.arguments['date']='2026-09-11'
        result=await self.run_booking(changed,net)
        self.assertEqual(result['error'],'booking_unresolved')
        await self.run_booking(self.booking(),net)
        self.assertEqual(net.requests[0][1]['bookingIntentId'],net.requests[1][1]['bookingIntentId'])
        self.assertTrue(handlers.is_booking_complete(self.sid))

    async def test_malformed_acceptance_keeps_intent_locked(self):
        for body in ([], None, 42, {'success': False}):
            with self.subTest(body=body):
                handlers._call_contexts[self.sid].pop('outstanding_booking', None)
                await self.run_booking(self.booking(), Network([Response(201, body)]))
                changed = self.booking()
                changed.arguments['date'] = '2026-09-11'
                result = await self.run_booking(changed, Network([]))
                self.assertEqual(result['error'], 'booking_unresolved')

    async def test_malformed_optional_metadata_does_not_erase_confirmed_booking(self):
        result = await self.run_booking(self.booking(), Network([Response(201, {
            'success': True, 'scheduled': True, 'jobId': 'J', 'sms': [], 'prices': 'invalid',
        })]))
        self.assertTrue(result['success'])
        self.assertEqual(handlers.get_booking_outcome(self.sid), 'scheduled')
        self.assertIn('Do not promise a text', result['message'])
    async def test_explicit_second_intent_same_call(self):
        net=Network([Response(201,{'success':True,'scheduled':True,'jobId':'J1'}),Response(201,{'success':True,'scheduled':True,'jobId':'J2'})])
        await self.run_booking(self.booking(),net)
        await self.run_booking(self.booking(booking_reference='second separately requested pickup'),net)
        self.assertNotEqual(net.requests[0][1]['bookingIntentId'],net.requests[1][1]['bookingIntentId'])
        self.assertEqual(len(handlers._call_contexts[self.sid]['bookings']),2)
    async def test_unverified_swap_rejected_before_network(self):
        params=self.booking(); params.arguments['type']='dumpster_swap'; params.arguments['phone']='+15559992222'
        result=await self.run_booking(params,Network([]))
        self.assertEqual(result['error'],'identity_not_verified')
    async def test_final_pickup_has_distinct_wire_type(self):
        params=self.booking(job_id='anchor',rental_address_confirmed=True);params.arguments['type']='dumpster_pickup'
        net=Network([Response(201,{'success':True,'scheduled':True,'jobId':'pickup'})])
        with patch.object(handlers,'resolve_job_id_from_lookup',return_value=('anchor',None)):
            result=await self.run_booking(params,net)
        self.assertEqual(net.requests[0][1]['serviceType'],'dumpster_pickup')
        self.assertEqual(net.requests[0][1]['verifiedJobId'],'anchor')
        self.assertIn('no replacement',result['message'])
    async def test_negative_mutations_preserve_state(self):
        handlers.mark_booking_outcome(self.sid,'scheduled',{'job_id':'J','appointment_date':'2026-09-10T13:00:00'})
        for handler,args in [(handlers.handle_cancel_appointment,{}),(handlers.handle_reschedule_appointment,{'new_date':'2026-09-11','new_time':'13:00-16:00'})]:
            with patch.object(handlers,'resolve_job_id_from_lookup',return_value=('J',None)),patch.object(handlers.aiohttp,'ClientSession',return_value=Network([Response(200,{'success':False})])):
                p=Params(phone=self.origin,**args);await handler(p)
            self.assertNotEqual(p.results[-1].get('success'),True)
            self.assertEqual(handlers.get_booking_log_state(self.sid)['appointment_date'],'2026-09-10T13:00:00')
    async def test_accepted_reschedule_then_cancel_updates_final_state(self):
        handlers.mark_booking_outcome(self.sid,'scheduled',{'job_id':'J','caller_name':'Old','appointment_date':'2026-09-10T13:00:00'})
        with patch.object(handlers,'resolve_job_id_from_lookup',return_value=('J',None)),patch.object(handlers.aiohttp,'ClientSession',return_value=Network([Response(200,{'success':True}),Response(200,{'success':True})])):
            await handlers.handle_reschedule_appointment(Params(phone=self.origin,new_date='2026-09-11',new_time='13:00-16:00'))
            self.assertEqual(handlers.get_booking_log_state(self.sid)['appointment_date'],'2026-09-11T13:00:00')
            await handlers.handle_cancel_appointment(Params(phone=self.origin))
        handlers.mark_caller_name(self.sid,'Corrected')
        logged=AsyncMock(return_value=True)
        with patch.object(bot,'_summarize_call_with_anthropic',AsyncMock(return_value='Cancelled.')),patch.object(bot,'log_call_to_dashboard',logged):
            await bot.run_post_call(call_id=self.sid,caller_number=self.origin,client_config=self.config,context=type('Context',(),{'messages':[]})(),call_start_time=datetime.now(ZoneInfo('America/Chicago'))-timedelta(seconds=60),timezone='America/Chicago')
        self.assertEqual(logged.await_args.kwargs['outcome'],'cancelled')
        self.assertEqual(logged.await_args.kwargs['appointment_date'],'')
        self.assertEqual(logged.await_args.kwargs['caller_name'],'Corrected')
    async def test_callback_alternate_number_and_invalid_success(self):
        net=Network([Response(201,{'success':False}),Response(201,{'success':True,'callbackTaskId':'T','callbackDueAt':'2026-09-10T13:00:00-05:00'})])
        with patch.object(handlers.aiohttp,'ClientSession',return_value=net):
            await handlers.handle_schedule_callback(Params(requested_time='2026-09-10T13:00:00',caller_phone='+15552220000'))
            self.assertFalse(handlers.get_transfer_state(self.sid)['callback_requested'])
            await handlers.handle_schedule_callback(Params(requested_time='2026-09-10T13:00:00',caller_phone='+15552220000'))
        state=handlers.get_transfer_state(self.sid)
        self.assertEqual(state['callback_phone'],'+15552220000');self.assertEqual(state['callback_source'],'explicit')
        self.assertEqual(handlers._call_contexts[self.sid]['caller_number'],self.origin)
    async def test_422_does_not_promise_or_mark_callback(self):
        net=Network([Response(422,{'error':'outside business hours'}),Response(422,{'error':'outside business hours'})])
        with patch.object(handlers.aiohttp,'ClientSession',return_value=net):
            for _ in range(2):
                p=Params(requested_time='2026-09-10T13:00:00');await handlers.handle_schedule_callback(p)
        self.assertIn('could not be arranged',p.results[-1]['message'])
        self.assertFalse(handlers.get_transfer_state(self.sid)['callback_requested'])
    async def test_after_hours_request_never_claims_unacknowledged_callback(self):
        with patch.object(handlers,'evaluate_current_business_hours',return_value=False), patch.object(handlers,'log_call_to_dashboard',AsyncMock()) as log:
            p=Params(reason='need a person');await handlers.handle_transfer_to_human(p)
        self.assertIn("isn't confirmed",p.results[-1]['say'])
        self.assertNotIn('already on the callback queue',p.results[-1]['note'])
        log.assert_not_awaited()
    async def test_promo_service_scope(self):
        net=Network([Response(200,{'valid':True,'discountType':'percentage','discountValue':10})])
        with patch.object(handlers.aiohttp,'ClientSession',return_value=net):
            p=Params(code='SAVE',service_type='dumpster');await handlers.handle_validate_promo_code(p)
        self.assertEqual(net.requests[0][1]['serviceType'],'dumpster')
        self.assertIn('provisional',p.results[-1]['message'])
    async def test_redirect_negative_200(self):
        with patch.object(dashboard_redirect.aiohttp,'ClientSession',return_value=Network([Response(200,{'success':False})])):
            result=await dashboard_redirect.redirect_live_call_via_dashboard(client_config={},client_id='org',call_sid='CA',reason='help',origin='phone_agent_ai_transfer',dashboard_url='https://fixture.invalid',platform_api_key='fixture')
        self.assertFalse(result.ok)
    async def test_cancel_survives_parent_cancellation(self):
        started, release = asyncio.Event(), asyncio.Event()
        class DelayedNetwork(Network):
            async def post(inner, url, **kwargs):
                started.set()
                await release.wait()
                return Response(200, {'success': True})
        handlers.mark_booking_outcome(self.sid,'scheduled',{'job_id':'J','appointment_date':'2026-09-10T13:00:00'})
        with patch.object(handlers,'resolve_job_id_from_lookup',return_value=('J',None)), patch.object(handlers.aiohttp,'ClientSession',return_value=DelayedNetwork([])):
            parent=asyncio.create_task(handlers.handle_cancel_appointment(Params(phone=self.origin)))
            await started.wait()
            parent.cancel()
            with self.assertRaises(asyncio.CancelledError): await parent
            pending=handlers.get_inflight_tasks(self.sid)
            self.assertEqual(len(pending),1)
            release.set()
            await asyncio.gather(*pending)
        self.assertEqual(handlers.get_booking_outcome(self.sid),'cancelled')
        self.assertFalse(handlers.is_booking_complete(self.sid))

    async def test_dumpster_only_rejects_junk_at_handler(self):
        self.config['companyMode']='dumpster_rental'
        result=await self.run_booking(self.booking(),Network([]))
        self.assertEqual(result['error'],'unsupported_service')


class PromptTests(unittest.TestCase):
    def test_modes_disclosure_extension_and_precise_hours(self):
        base={'companyName':'Fixture','businessDays':[1,2,3,4,5], 'businessHours':{'mon':{'open':'08:30','close':'16:30'},'tue':{'open':'10:00','close':'14:15'}},'dumpsterRentalsEnabled':True}
        for mode in ('junk_removal','dumpster_rental','both'):
            rendered=build_system_prompt({**base,'companyMode':mode})
            self.assertIn('AI receptionist',rendered)
            self.assertNotIn('Never reveal you are AI',rendered)
            self.assertNotIn('keep it as long as you need',rendered)
            self.assertIn('8:30 AM to 4:30 PM',rendered)
            self.assertIn('10 AM to 2:15 PM',rendered)
            if mode=='dumpster_rental':
                self.assertNotIn('ALWAYS use type: "pickup"',rendered)
                self.assertNotIn('Payment question (junk removal)',rendered)
                self.assertNotIn('redirect to junk removal pickup',rendered)
        schema=bot._create_booking_schema(True,False)
        self.assertNotIn('pickup',schema.properties['type']['enum'])


class SilenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_nudge_speech_does_not_restart_nudges_forever(self):
        clock = 0
        speaking_until = 0
        spoken, cancelled = [], []
        async def sleep(seconds):
            nonlocal clock
            clock += seconds
            if clock > 150:
                raise RuntimeError('watchdog failed to finish')
        async def speak(text):
            nonlocal speaking_until
            spoken.append(text)
            speaking_until = clock + 7
        async def cancel():
            cancelled.append(clock)
        await watch_silence(activity=lambda: 0, busy=lambda: clock < speaking_until,
            booked=lambda: False, speak=speak, cancel=cancel, now=lambda: clock, sleep=sleep)
        self.assertTrue(cancelled)
        self.assertEqual(len(spoken), 2)

    async def scenario(self, booked, resume=False, in_flight=False):
        clock=0; activity=0; cancelled=[]; spoken=[]; delays=[]
        async def sleep(seconds):
            nonlocal clock,activity
            clock+=seconds;delays.append(seconds)
            if seconds==8 and resume and not cancelled:
                activity+=1
                if len([x for x in delays if x==8])>1: raise RuntimeError('stop')
            if clock>120: raise RuntimeError('stop')
        async def speak(text): spoken.append(text)
        async def cancel(): cancelled.append(clock)
        try:
            await watch_silence(activity=lambda:activity,busy=lambda:in_flight,booked=lambda:booked,speak=speak,cancel=cancel,now=lambda:clock,sleep=sleep)
        except RuntimeError: pass
        return cancelled,spoken
    async def test_resumed_speech_prevents_goodbye_cancel_in_both_phases(self):
        for phase in (False,True):
            cancelled,spoken=await self.scenario(phase,resume=True)
            self.assertFalse(cancelled);self.assertTrue(spoken)
    async def test_midcall_silence_is_recovered_and_inflight_is_protected(self):
        cancelled,spoken=await self.scenario(False)
        self.assertTrue(cancelled);self.assertEqual(len(spoken),2)
        cancelled,spoken=await self.scenario(False,in_flight=True)
        self.assertFalse(cancelled);self.assertFalse(spoken)
