"""Native audio/embedding routing, batch integrity, caching, and terminal refusals."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import numpy as np

from evaluation.clients import Client, ServiceError
from evaluation.io_utils import write_json
from methods.longemo.audio import audio_client_for
from methods.longemo.common import LoggedClient
from methods.longemo.embeddings import APIEncoder
from methods.longemo.runner import parser


class NativeGeminiTest(unittest.TestCase):
    def test_audio_uses_native_key_and_low_thinking(self):
        with tempfile.TemporaryDirectory() as tmp:
            key = Path(tmp)/'keys.json'
            write_json(key,{'GEMINI_API_KEY':'native-secret','OPENROUTER_API_KEY':'different-secret'})
            args=parser().parse_args(['build','--data-path','unused','--videos-dir','unused','--output-dir',tmp,
                '--model','gpt-6-astra','--with-audio','--credential-file',str(key),
                '--audio-model','gemini-3.8-flash','--audio-base-url','https://www.blackaicoding.com/v1beta'])
            client=audio_client_for(args)
            self.assertEqual(client.api_key,'native-secret')
            self.assertEqual(client.api_format,'gemini')
            payload=client.payload([{'role':'user','content':[{'type':'input_audio','input_audio':{'data':'AA==','format':'wav'}}]}])
            self.assertEqual(payload['generationConfig']['thinkingConfig']['thinkingLevel'],'low')
            self.assertEqual(payload['contents'][0]['parts'][0]['inline_data']['mime_type'],'audio/wav')
            self.assertNotIn('secret',json.dumps(client.configuration()))

    def test_native_embedding_separate_documents_order_cache_and_query_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            encoder=APIEncoder('secret',tmp,model='gemini-embedding-2',base_url='https://www.blackaicoding.com/v1beta',api_format='gemini',dimension=3)
            payloads=[]
            def send(req,**kwargs):
                self.assertTrue(req.full_url.endswith('/models/gemini-embedding-2:batchEmbedContents'))
                self.assertEqual(req.headers['X-goog-api-key'],'secret')
                self.assertNotIn('Authorization',req.headers)
                p=json.loads(req.data);payloads.append(p)
                for x in p['requests']:
                    self.assertEqual(len(x['content']['parts']),1)
                    self.assertEqual(x['model'],'models/gemini-embedding-2')
                    self.assertEqual(x['outputDimensionality'],3)
                vectors=[[3.,0.,0.],[0.,4.,0.]][:len(p['requests'])]
                return io.BytesIO(json.dumps({'embeddings':[{'values':x} for x in vectors]}).encode())
            with patch('urllib.request.urlopen',side_effect=send) as requests:
                result=encoder.encode(['calm','angry'])
                np.testing.assert_allclose(result,[[1,0,0],[0,1,0]])
                np.testing.assert_array_equal(encoder.encode(['calm','angry']),result)
                self.assertEqual(requests.call_count,1)
                encoder.encode(['calm'],query=True)
                self.assertEqual(requests.call_count,2)
            self.assertTrue(payloads[0]['requests'][0]['content']['parts'][0]['text'].startswith('title: Emotional event | text: '))
            self.assertTrue(payloads[1]['requests'][0]['content']['parts'][0]['text'].startswith('task: search result | query: '))
            self.assertNotIn('secret',''.join(p.read_text() for p in Path(tmp).glob('*.json*')))

    def test_native_batch_cannot_silently_aggregate_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            encoder=APIEncoder('secret',tmp,model='gemini-embedding-2',base_url='https://www.blackaicoding.com/v1beta',api_format='gemini',dimension=3,tries=1)
            reply=io.BytesIO(b'{"embeddings":[{"values":[1,0,0]}]}')
            with patch('urllib.request.urlopen',return_value=reply):
                with self.assertRaisesRegex(RuntimeError,'embedding API failed'):
                    encoder.encode(['first','second'])

    def test_embedding_permission_failure_has_one_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            encoder=APIEncoder('secret',tmp,model='gemini-embedding-2',base_url='https://www.blackaicoding.com/v1beta',api_format='gemini')
            exc=HTTPError('https://www.blackaicoding.com',404,'Not found',{},io.BytesIO(b'private provider message'))
            with patch('urllib.request.urlopen',side_effect=exc) as requests:
                with self.assertRaises(ServiceError):encoder.encode(['calm'])
            self.assertEqual(requests.call_count,1)
            self.assertNotIn('private',Path(tmp,'calls.jsonl').read_text())

    def test_native_safety_refusal_is_logged_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            client=Client('gemini-3.8-flash','https://www.blackaicoding.com/v1beta','gemini','secret')
            reply=io.BytesIO(b'{"promptFeedback":{"blockReason":"SAFETY"}}')
            with patch('urllib.request.urlopen',return_value=reply) as requests:
                with self.assertRaisesRegex(RuntimeError,'after 1 attempts'):
                    LoggedClient(client,Path(tmp)/'calls.jsonl',3).call([{'role':'user','content':'input'}],purpose='probe')
            self.assertEqual(requests.call_count,1)
            record=json.loads(Path(tmp,'calls.jsonl').read_text())
            self.assertEqual(record['http_status'],200)
            self.assertEqual(record['service_error_code'],'content_filter')

    def test_response_observer_preserves_parser_behavior_and_excludes_content(self):
        from experiments.zyf.native_worker import observed_parser
        from evaluation.inference.adapters.gemini import parse_response
        with tempfile.TemporaryDirectory() as tmp:
            ledger=Path(tmp)/'responses.jsonl'
            client=Client('gemini-3.8-flash','https://www.blackaicoding.com/v1beta','gemini','secret-key')
            raw={'modelVersion':'gemini-3.8-flash','candidates':[{'finishReason':'MAX_TOKENS',
                'content':{'parts':[{'text':'private-response-text'}]}}], 'usageMetadata':{'totalTokenCount':123}}
            parser=observed_parser(parse_response,ledger)
            with self.assertRaisesRegex(RuntimeError,'MAX_TOKENS'):parser(client,raw)
            raw['candidates'][0]['finishReason']='STOP'
            self.assertEqual(parser(client,raw),parse_response(client,raw))
            content=ledger.read_text()
            self.assertNotIn('private-response-text',content)
            self.assertNotIn('secret-key',content)
            rows=[json.loads(line) for line in content.splitlines()]
            self.assertEqual(rows[0]['finish_reasons'],['MAX_TOKENS'])
            self.assertEqual(rows[0]['usage']['totalTokenCount'],123)


if __name__=='__main__':unittest.main()
