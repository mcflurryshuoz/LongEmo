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

    def test_gemini_changes_only_perception_and_requires_new_run(self):
        self._full_run(full.GEMINI_PERCEPTION)

    def _full_run(self, perception):
        data=self.root/'data'
        write_json(data/'questions.json',[question('Q1'),question('Q2')])
        write_json(data/'manifest.json',{'question_count':2,'video_count':1,'revision':'fixed','pilot_question_ids':[]})
        key=self.root/'private.json'
        write_json(key,{'OPENROUTER_API_KEY':'unit-test-placeholder'})
        commands=[]
        def run(cmd, **kwargs):
            commands.append(cmd)
            gemini = perception == full.GEMINI_PERCEPTION and cmd[3:5] == ['methods.longemo','build']
            self.assertEqual(cmd[cmd.index('--model')+1],perception if gemini else full.MODEL)
            self.assertEqual(cmd[cmd.index('--base-url')+1],full.OPENROUTER_URL if gemini else full.BASE_URL)
            self.assertNotIn('MODEL_API_KEY',kwargs['env'])
            if gemini:
                self.assertEqual(full.read(cmd[cmd.index('--config')+1]),{'reasoning':{'effort':'medium'}})
                self.assertEqual(cmd[cmd.index('--temperature')+1],'1')
            output=Path(cmd[cmd.index('--output-dir')+1])
            if cmd[3]=='methods.longemo' and cmd[4]=='answer':
                write_records(output/'predictions.jsonl',[{'question_id':q,'video_id':'V1','status':'ok','prediction':'answer'} for q in ('Q1','Q2')])
            if cmd[3]=='evaluation.eval':
                write_records(output/'run_1'/'scores.jsonl',[score('Q1',0),score('Q2',4)])
            return SimpleNamespace(returncode=0)
        def balance(*args, **kwargs):
            return io.BytesIO(b'{"data":{"total_credits":10,"total_usage":0}}')
        args=['--data-root',str(data),'--output-dir',str(self.root/'out'),'--credential-file',str(key),
            '--embedding-cache-dir',str(self.root/'cache'),'--perception-model',perception,
            '--startup-videos','1','--execute']
        with patch.object(full,'code_hash',return_value=full.SOURCE_HASH), patch.object(full,'git_revision',return_value='test'), \
             patch.object(full,'inventory',return_value={'complete':True,'video_count':1,'videos':{'V1':{'duration_seconds':20}}}), \
             patch.object(full,'urlopen',side_effect=balance), patch.object(full.subprocess,'run',side_effect=run), \
             patch.object(az,'access_token',return_value='secret'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(full.main(args),0)
        self.assertEqual(len(commands),3)
        self.assertEqual(full.read(self.root/'out'/'metrics.json')['overall_unweighted']['coverage'],1)
        config=full.read(self.root/'out'/'experiment_manifest.json')['configuration']
        self.assertEqual(config['model'],full.MODEL)
        self.assertEqual(config['judge_model'],full.MODEL)
        self.assertEqual(config['audio_model'],full.AUDIO)
        if perception == full.GEMINI_PERCEPTION:
            self.assertEqual(config['perception_model'],perception)
            incompatible={**config,'perception_model':full.MODEL}
            with self.assertRaisesRegex(ValueError,'semantic'):
                full.execution_manifest(self.root/'out'/'experiment_manifest.json',incompatible)
        else:
            self.assertNotIn('perception_model',config)
            self.assertEqual(config['protocol'],'azure-full-episode-graph-v1')


if __name__=='__main__':
    unittest.main()
