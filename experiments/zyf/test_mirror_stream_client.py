import json
import time
import unittest
from unittest.mock import patch

from evaluation.clients import ServiceError
from experiments.zyf.mirror_stream_client import StreamClient
from experiments.zyf.test_mirror_stream_probe import response


class FakeResponse:
    status = 200
    def __init__(self, rows):
        self.rows=rows
        self.headers=self
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def get_content_type(self):return 'text/event-stream'
    def __iter__(self):
        for row in self.rows:
            yield ('data: '+json.dumps(row)+'\n').encode()
            yield b'\n'


class ClientTests(unittest.TestCase):
    def client(self):return StreamClient('claude-opus-5','https://api.aicodemirror.ai/api/claudecode/v1','anthropic','',240,8192,None,{'stream':True})
    def test_only_transport_field_added_and_final_content_reassembled(self):
        client=self.client();messages=[{'role':'user','content':'unchanged prompt'}]
        with patch('experiments.zyf.mirror_stream_client.request.urlopen',return_value=FakeResponse(response())) as call:
            result=client.generate(messages)
        payload=json.loads(call.call_args.args[0].data)
        self.assertEqual(payload,{'model':'claude-opus-5','messages':messages,'max_tokens':8192,'stream':True})
        self.assertEqual(result['content'],'{"a":1}')
        self.assertEqual(call.call_count,1)
    def test_refusal_never_returned_as_normal_answer(self):
        rows=response();rows[-2]['delta']['stop_reason']='refusal'
        with patch('experiments.zyf.mirror_stream_client.request.urlopen',return_value=FakeResponse(rows)):
            with self.assertRaises(ServiceError) as exc:self.client().generate([{'role':'user','content':'x'}])
        self.assertEqual(exc.exception.code,'content_filter')
        self.assertFalse(exc.exception.retryable)
    def test_truncation_is_not_implicitly_retried(self):
        with patch('experiments.zyf.mirror_stream_client.request.urlopen',return_value=FakeResponse(response()[:-1])) as call:
            with self.assertRaises(ValueError):self.client().generate([{'role':'user','content':'x'}])
        self.assertEqual(call.call_count,1)


if __name__=='__main__':unittest.main()
