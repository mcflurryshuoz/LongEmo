"""Azure routing, quota coordination, fail-fast errors, and full-run wiring."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

from evaluation import azure_transport as az
from evaluation.clients import Client
from evaluation.io_utils import write_json, write_records
from experiments.zyf import azure_benchmark as full
from experiments.zyf.test_full_benchmark import question, score


class AzureTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root/'limits.json'
        write_json(self.config, {'tokens_per_minute': 1000, 'requests_per_minute': 10, 'max_concurrent': 1})
        self.env = patch.dict(os.environ, AZURE_RATE_LIMIT_DB=str(self.root/'quota.sqlite'),
            AZURE_RATE_LIMIT_CONFIG=str(self.config), AZURE_TRANSPORT_LEDGER_DIR=str(self.root/'ledger'))
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_shared_slots_refund_and_live_limits(self):
        a, b = az.RateLimiter(), az.RateLimiter()
        rid, _, _ = a.acquire(700)
        with self.assertRaises(TimeoutError):
            b.acquire(100, timeout=0)
        a.release(rid, 100)
        rid2, _, _ = b.acquire(800)
        b.release(rid2)
        with self.assertRaises(TimeoutError):
            a.acquire(101, timeout=0)
        write_json(self.config, {'tokens_per_minute': 2000, 'requests_per_minute': 10, 'max_concurrent': 2})
        rid3, _, limits = a.acquire(101, timeout=0)
        self.assertEqual(limits['tokens_per_minute'], 2000)
        a.release(rid3)

    def test_image_base64_is_not_text_tokens(self):
        small={'messages':[{'type':'image_url','image_url':{'url':'data:abcd'}}], 'max_completion_tokens':100}
        large={'messages':[{'type':'image_url','image_url':{'url':'data:'+'x'*100000}}], 'max_completion_tokens':100}
        self.assertEqual(az.token_estimate(small), az.token_estimate(large))

    def test_azure_routing_auth_and_raw_usage(self):
        client = Client('gpt-6-astra', full.BASE_URL, max_tokens=50, temperature=None,
            options={'max_completion_tokens':50, 'reasoning_effort':'medium'})
        response = io.BytesIO(json.dumps({'model':'version-test', 'choices':[{'message':{'content':'OK'}}],
            'usage':{'prompt_tokens':10,'completion_tokens':2,'total_tokens':12}}).encode())
        response.headers = {'x-ratelimit-limit-tokens':'1000000'}
        def send(req, **kwargs):
            self.assertIn('/openai/deployments/gpt-6-astra/chat/completions?api-version=', req.full_url)
            self.assertEqual(req.headers['Authorization'], 'Bearer unit-test-secret')
            payload=json.loads(req.data)
            self.assertEqual(payload['reasoning_effort'], 'medium')
            self.assertNotIn('temperature',payload)
            return response
        with patch.object(az, 'access_token', return_value='unit-test-secret'), patch.object(az.request, 'urlopen', side_effect=send):
            result=client.generate([{'role':'user','content':'OK'}])
        self.assertEqual(result['content'],'OK')
        ledger=''.join(p.read_text() for p in (self.root/'ledger').glob('*.jsonl'))
        self.assertNotIn('unit-test-secret',ledger)
        self.assertEqual(json.loads(ledger)['usage']['total_tokens'],12)
        self.assertEqual(client.configuration()['transport']['provider'],'azure')

    def test_permission_error_has_no_retries_or_body_leak(self):
        client=Client('gpt-6-astra', full.BASE_URL, max_tokens=50, temperature=None, options={'max_completion_tokens':50})
        failure=HTTPError(full.BASE_URL,403,'Forbidden',{},io.BytesIO(b'{"error":{"code":"PermissionDenied","message":"private prompt"}}'))
        with patch.object(az, 'access_token', return_value='secret'), patch.object(az.request, 'urlopen', side_effect=failure) as send:
            with self.assertRaises(az.AzureError) as caught:
                client.generate([{'role':'user','content':'OK'}])
        self.assertFalse(caught.exception.retryable)
        self.assertEqual(send.call_count,1)
        self.assertNotIn('private', str(caught.exception))
        with az.RateLimiter().connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM reservations WHERE active=1').fetchone()[0],0)

    def test_429_uses_shared_cooldown_and_records_each_attempt(self):
        client=Client('gpt-6-astra', full.BASE_URL, max_tokens=50, temperature=None, options={'max_completion_tokens':50})
        response=io.BytesIO(json.dumps({'choices':[{'message':{'content':'OK'}}], 'usage':{'total_tokens':12}}).encode())
        response.headers={}
        failure=HTTPError(full.BASE_URL,429,'busy',{'retry-after':'3'},io.BytesIO(b'{"error":{"code":"RateLimitReached"}}'))
        with patch.object(az,'access_token',return_value='secret'), \
             patch.object(az.request,'urlopen',side_effect=[failure,response]) as send, \
             patch.object(az.RateLimiter,'pause') as pause, patch.object(az.time,'sleep'):
            self.assertEqual(client.generate([{'role':'user','content':'OK'}])['content'],'OK')
        self.assertEqual(send.call_count,2)
        self.assertGreaterEqual(pause.call_args.args[0],3)
        records=[json.loads(line) for p in (self.root/'ledger').glob('*.jsonl') for line in p.read_text().splitlines()]
        self.assertEqual([r['http_status'] for r in records],[429,200])

    def test_scheduler_revision_preserves_semantics_and_original_manifest(self):
        p=self.root/'experiment_manifest.json'
        first={'code_hash':'fixed','model':'same','orchestrator_sha256':'old','git_revision':'a'}
        full.execution_manifest(p,first)
        original=p.read_bytes()
        second={**first,'orchestrator_sha256':'new','git_revision':'b'}
        full.execution_manifest(p,second)
        self.assertEqual(p.read_bytes(),original)
        self.assertTrue((self.root/'execution_revisions/new/manifest.json').exists())
        with self.assertRaisesRegex(ValueError,'semantic'):
            full.execution_manifest(p,{**second,'code_hash':'changed'})
        with self.assertRaisesRegex(ValueError,'semantic'):
            full.execution_manifest(p,{**second,'model':'different'})

    def test_policy_block_is_not_a_retryable_video(self):
        memory=self.root/'memory/V1';folder=self.root/'videos/V1'
        write_records(memory/'calls.jsonl',[{'status':'error','http_status':400,
            'service_error_code':'content_policy_violation','purpose':'perception:V1:W1'}])
        blocked=full.policy_rejection(folder,memory)
        self.assertEqual(blocked['purpose'],'perception:V1:W1')
        self.assertIsNone(full.policy_rejection(self.root/'videos/V2',self.root/'memory/V2'))

    def test_full_graph_and_official_judge_both_use_azure(self):
        self._full_run(full.MODEL)

    def test_428_uses_latest_attempt_and_keeps_other_error_classes(self):
        folder=self.root/'videos/V1';memory=self.root/'memory/V1'
        path=memory/'audio/calls.jsonl'
        failure={'purpose':'audio_observer:W1','status':'error','http_status':428,'service_error_code':'428'}
        write_records(path,[failure])
        self.assertEqual(full.recorded_video_block(folder,memory)['status'],'blocked_request_precondition')
        write_records(path,[failure,{'purpose':'audio_observer:W1','status':'ok'}])
        self.assertIsNone(full.recorded_video_block(folder,memory))
        for code in (401,402,403,429,500):
            write_records(path,[{**failure,'http_status':code,'service_error_code':str(code)}])
            self.assertIsNone(full.recorded_video_block(folder,memory))
        write_records(path,[failure,{**failure,'purpose':'audio_observer:W2','service_error_code':'content_filter'}])
        self.assertEqual(full.recorded_video_block(folder,memory)['status'],'blocked_input_policy')

    def test_three_428_videos_do_not_stop_others_or_shrink_denominator(self):
        data=self.root/'data';out=self.root/'out';key=self.root/'private.json'
        questions=[{**question('Q'+str(i)),'video_id':'V'+str(i)} for i in range(1,7)]
        write_json(data/'questions.json',questions)
        write_json(data/'manifest.json',{'question_count':6,'video_count':6,'revision':'fixed','pilot_question_ids':[]})
        write_json(key,{'OPENROUTER_API_KEY':'embedding-placeholder','MODEL_API_KEY':'matrix-placeholder'})
        failure={'purpose':'audio_observer:W1','status':'error','http_status':428,'service_error_code':'428'}
        write_records(out/'memory/V5/audio/calls.jsonl',[failure])
        write_records(out/'memory/V6/audio/calls.jsonl',[{**failure,'service_error_code':'content_filter'}])
        builds=[]
        def run(cmd,**kwargs):
            subset=full.read(cmd[cmd.index('--data-path')+1]);q=subset[0];vid=q['video_id']
            destination=Path(cmd[cmd.index('--output-dir')+1])
            if cmd[4]=='build':
                builds.append(vid)
                if vid in ('V1','V2','V3'):
                    write_records(out/'memory'/vid/'audio/calls.jsonl',[failure])
                    return SimpleNamespace(returncode=1)
            elif cmd[4]=='answer':
                write_records(destination/'predictions.jsonl',[{'question_id':q['question_id'],'video_id':vid,'status':'ok','prediction':'answer'}])
            elif cmd[3]=='evaluation.eval':
                write_records(destination/'run_1/scores.jsonl',[{**score(q['question_id'],4),'video_id':vid}])
            return SimpleNamespace(returncode=0)
        inv={'complete':True,'video_count':6,'videos':{'V'+str(i):{'duration_seconds':20*i} for i in range(1,7)}}
        args=['--data-root',str(data),'--output-dir',str(out),'--credential-file',str(key),
              '--embedding-cache-dir',str(self.root/'cache'),'--perception-model',full.NATIVE_GEMINI,
              '--matrix-perception','--answer-base-url',full.MATRIX_URL,'--embedding-backend','gemini',
              '--video-workers','1','--video-attempts','2','--execute']
        with patch.object(full,'code_hash',return_value=full.SOURCE_HASH),patch.object(full,'git_revision',return_value='test'), \
             patch.object(full,'inventory',return_value=inv),patch.object(full.subprocess,'run',side_effect=run), \
             patch.object(full,'urlopen',side_effect=lambda *a,**k:io.BytesIO(b'{"data":{"total_credits":10,"total_usage":0}}')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertNotEqual(full.main(args),0)
        self.assertEqual(builds,['V1','V2','V3','V4'])
        state=full.read(out/'status.json')
        self.assertFalse(state['scheduling_paused'])
        self.assertEqual(state['videos']['V4']['status'],'complete')
        self.assertEqual(state['videos']['V5']['status'],'blocked_request_precondition')
        self.assertEqual(state['videos']['V6']['status'],'blocked_input_policy')
        metrics=full.read(out/'metrics.json')['overall_unweighted']
        self.assertEqual((metrics['n_scored'],metrics['n_total']),(1,6))

    def test_gemini_changes_only_perception_and_requires_new_run(self):
        self._full_run(full.GEMINI_PERCEPTION)

    def test_native_frontend_and_embeddings_never_use_openrouter(self):
        self._full_run(full.NATIVE_GEMINI)

    def test_native_memory_continues_when_embedding_is_not_enabled(self):
        self._full_run(full.NATIVE_GEMINI, defer=True)

    def test_native_deferred_queue_retries_startup_then_builds_remaining_videos(self):
        data=self.root/'data'
        questions=[{**question('Q'+str(i)), 'video_id':'V'+str(i)} for i in range(1,4)]
        write_json(data/'questions.json',questions)
        write_json(data/'manifest.json',{'question_count':3,'video_count':3,'revision':'fixed','pilot_question_ids':[]})
        key=self.root/'private.json'
        # Native execution must not even require an OpenRouter credential.
        write_json(key,{'GEMINI_API_KEY':'native-test-placeholder'})
        attempts=[]
        def run(cmd, **kwargs):
            self.assertEqual(cmd[3:5],['experiments.zyf.native_worker','build'])
            self.assertNotIn('OPENROUTER_API_KEY',kwargs['env'])
            vid=full.read(cmd[cmd.index('--data-path')+1])[0]['video_id']
            attempts.append(vid)
            return SimpleNamespace(returncode=1 if len(attempts)==1 else 0)
        inv={'complete':True,'video_count':3,'videos':{'V'+str(i):{'duration_seconds':20*i} for i in range(1,4)}}
        args=['--data-root',str(data),'--output-dir',str(self.root/'out'),'--credential-file',str(key),
            '--embedding-cache-dir',str(self.root/'cache'),'--perception-model',full.NATIVE_GEMINI,
            '--startup-videos','1','--video-workers','2','--defer-answers','--execute']
        with patch.object(full,'code_hash',return_value=full.SOURCE_HASH), patch.object(full,'git_revision',return_value='test'), \
             patch.object(full,'inventory',return_value=inv), patch.object(full.subprocess,'run',side_effect=run), \
             patch.object(full,'urlopen',side_effect=AssertionError('unexpected OpenRouter call')), \
             patch.object(az,'access_token',return_value='secret'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(full.main(args),3)
        self.assertEqual(attempts[:2],['V1','V1'])
        self.assertCountEqual(attempts[2:],['V2','V3'])
        status=full.read(self.root/'out/status.json')
        self.assertEqual(status['status'],'partial_embedding_service')
        self.assertEqual(status['queued_videos'],0)
        self.assertEqual(len(status['videos']),3)
        self.assertEqual(status['n_scored'],0)

    def test_matrix_judge_and_blackai_perception_use_separate_credentials(self):
        self._full_run(full.NATIVE_GEMINI, matrix=True)

    def test_matrix_audio_uses_chat_and_explicit_matrix_credential(self):
        from methods.longemo.audio import audio_client_for
        credentials=self.root/'audio.json'
        write_json(credentials,{'MODEL_API_KEY':'matrix-placeholder','GEMINI_API_KEY':'wrong-provider'})
        client=audio_client_for(SimpleNamespace(model=full.NATIVE_GEMINI, audio_model=full.NATIVE_GEMINI,
            audio_base_url=full.MATRIX_URL, credential_file=credentials, with_audio=True))
        self.assertEqual(client.api_format,'chat')
        self.assertEqual(client.api_key,'matrix-placeholder')
        self.assertEqual(client.options,{'reasoning_effort':'low'})

    def test_matrix_multimodal_frontend_waits_for_data_before_admission(self):
        self._full_run(full.NATIVE_GEMINI, matrix=True, matrix_frontend=True, wait_data=True)

    def _full_run(self, perception, defer=False, matrix=False, matrix_frontend=False, wait_data=False):
        data=self.root/'data'
        write_json(data/'questions.json',[question('Q1'),question('Q2')])
        write_json(data/'manifest.json',{'question_count':2,'video_count':1,'revision':'fixed','pilot_question_ids':[]})
        key=self.root/'private.json'
        write_json(key,{'OPENROUTER_API_KEY':'unit-test-placeholder','GEMINI_API_KEY':'native-test-placeholder'})
        perception_key=self.root/'perception.json'
        write_json(perception_key, {'GEMINI_API_KEY':'native-test-placeholder'})
        if matrix:
            write_json(key, {**full.read(key),'MODEL_API_KEY':'matrix-test-placeholder'})
        answer_base='https://matrixllm.alipay.com/v1' if matrix else full.BASE_URL
        commands=[]
        def run(cmd, **kwargs):
            commands.append(cmd)
            native = perception == full.NATIVE_GEMINI and not matrix_frontend
            gemini = perception in (full.GEMINI_PERCEPTION,full.NATIVE_GEMINI) and cmd[4] == 'build'
            if native and gemini:
                self.assertEqual(cmd[3],'experiments.zyf.native_worker')
                self.assertEqual(cmd[cmd.index('--max-tokens')+1],'32768')
            elif matrix_frontend and gemini:
                self.assertEqual(cmd[3],'methods.longemo')
                self.assertEqual(cmd[cmd.index('--max-tokens')+1],'32768')
            else:
                self.assertEqual(cmd[cmd.index('--max-tokens')+1],'8192')
            self.assertEqual(cmd[cmd.index('--model')+1],perception if gemini else full.MODEL)
            self.assertEqual(cmd[cmd.index('--base-url')+1],(full.MATRIX_URL if matrix_frontend else full.BLACKAI_URL if native else full.OPENROUTER_URL) if gemini else answer_base)
            if matrix:
                self.assertEqual(kwargs['env']['MODEL_API_KEY'],'matrix-test-placeholder')
                if gemini:
                    self.assertEqual(cmd[cmd.index('--credential-file')+1],str(key if matrix_frontend else perception_key))
                elif cmd[3] != 'evaluation.eval':
                    self.assertEqual(cmd[cmd.index('--credential-file')+1],str(key))
            else:
                self.assertNotIn('MODEL_API_KEY',kwargs['env'])
            if gemini:
                settings = {'generationConfig':{'thinkingConfig':{'thinkingLevel':'medium'}}} if native else {'reasoning_effort':'medium'} if matrix_frontend else {'reasoning':{'effort':'medium'}}
                self.assertEqual(full.read(cmd[cmd.index('--config')+1]),settings)
                self.assertEqual(cmd[cmd.index('--temperature')+1],'1')
            output=Path(cmd[cmd.index('--output-dir')+1])
            if cmd[3]=='methods.longemo' and cmd[4]=='answer':
                if matrix:
                    self.assertEqual(cmd[cmd.index('--embedding-backend')+1],'gemini')
                    self.assertEqual(cmd[cmd.index('--embedding-model')+1],full.EMBEDDING)
                    self.assertEqual(cmd[cmd.index('--embedding-base-url')+1],full.OPENROUTER_URL)
                elif native:
                    self.assertEqual(cmd[cmd.index('--embedding-backend')+1],'gemini-native')
                    self.assertEqual(cmd[cmd.index('--embedding-model')+1],full.NATIVE_EMBEDDING)
                    self.assertEqual(cmd[cmd.index('--embedding-base-url')+1],full.BLACKAI_URL)
                    self.assertEqual(cmd[cmd.index('--audio-base-url')+1],full.BLACKAI_URL)
                write_records(output/'predictions.jsonl',[{'question_id':q,'video_id':'V1','status':'ok','prediction':'answer'} for q in ('Q1','Q2')])
            if cmd[3]=='evaluation.eval':
                write_records(output/'run_1'/'scores.jsonl',[score('Q1',0),score('Q2',4)])
            return SimpleNamespace(returncode=0)
        def balance(*args, **kwargs):
            if perception == full.NATIVE_GEMINI and not matrix:
                raise AssertionError('native Gemini experiment must not call OpenRouter')
            return io.BytesIO(b'{"data":{"total_credits":10,"total_usage":0}}')
        args=['--data-root',str(data),'--output-dir',str(self.root/'out'),'--credential-file',str(key),
            '--embedding-cache-dir',str(self.root/'cache'),'--perception-model',perception,
            '--startup-videos','1','--execute']
        if defer:
            args += ['--defer-answers']
        if matrix:
            args += ['--answer-base-url',answer_base,'--embedding-backend','gemini']
            args += ['--matrix-perception'] if matrix_frontend else ['--perception-credential-file',str(perception_key)]
        complete_inventory={'complete':True,'video_count':1,'videos':{'V1':{'duration_seconds':20}}}
        inventories=[complete_inventory]
        if wait_data:
            args += ['--wait-for-data']
            inventories.insert(0,{'complete':False,'video_count':0,'videos':{},'missing_subtitles':[]})
        with patch.object(full,'code_hash',return_value=full.SOURCE_HASH), patch.object(full,'git_revision',return_value='test'), \
             patch.object(full,'inventory',side_effect=inventories), \
             patch.object(full.time,'sleep'), patch.object(full.time,'monotonic',side_effect=__import__('itertools').count(0,31)), \
             patch.object(full,'urlopen',side_effect=balance), patch.object(full.subprocess,'run',side_effect=run), \
             patch.object(az,'access_token',side_effect=AssertionError('unexpected Azure login') if matrix else None,return_value='secret'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(full.main(args),3 if defer else 0)
        self.assertEqual(len(commands),1 if defer else 3)
        self.assertEqual(full.read(self.root/'out'/'metrics.json')['overall_unweighted']['coverage'],0 if defer else 1)
        if defer:
            self.assertEqual(full.read(self.root/'out/status.json')['status'],'partial_embedding_service')
        config=full.read(self.root/'out'/'experiment_manifest.json')['configuration']
        self.assertEqual(config['model'],full.MODEL)
        self.assertEqual(config['judge_model'],full.MODEL)
        self.assertEqual(config['base_url'],answer_base)
        if matrix:
            self.assertNotIn('azure_transport',config)
            self.assertEqual(config['embedding_model'],full.EMBEDDING)
            self.assertEqual(config['embedding_base_url'],full.OPENROUTER_URL)
        self.assertEqual(config['audio_model'],full.NATIVE_GEMINI if perception == full.NATIVE_GEMINI else full.AUDIO)
        if perception in (full.GEMINI_PERCEPTION,full.NATIVE_GEMINI):
            self.assertEqual(config['perception_model'],perception)
            incompatible={**config,'perception_model':full.MODEL}
            with self.assertRaisesRegex(ValueError,'semantic'):
                full.execution_manifest(self.root/'out'/'experiment_manifest.json',incompatible)
        else:
            self.assertNotIn('perception_model',config)
            self.assertEqual(config['protocol'],'azure-full-episode-graph-v1')


if __name__=='__main__':
    unittest.main()
