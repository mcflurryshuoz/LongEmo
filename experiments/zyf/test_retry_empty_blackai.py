import json
from pathlib import Path
import tempfile
import unittest

from evaluation.clients import Client
from evaluation.inference.adapters.gemini import parse_response
from evaluation.io_utils import write_json
from experiments.zyf.native_worker import observed_parser
from experiments.zyf.retry_empty_blackai import clone_checkpoint


class RetryEmptyTest(unittest.TestCase):
    def test_empty_response_keeps_behavior_and_records_only_safe_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'responses.jsonl'
            client=Client('gemini-3.8-flash','https://www.blackaicoding.com/v1beta','gemini','secret-key')
            raw={'candidates':[],'usageMetadata':{'promptTokenCount':22},
                 'error':{'code':503,'status':'UNAVAILABLE','message':'private prompt text'}}
            with self.assertRaisesRegex(RuntimeError,'no answer candidate'):
                observed_parser(parse_response,path)(client,raw)
            r=json.loads(path.read_text())
            self.assertTrue(r['empty_candidates']);self.assertEqual(r['error_code'],503)
            self.assertNotIn('private prompt text',path.read_text());self.assertNotIn('secret-key',path.read_text())

    def test_checkpoint_copy_preserves_source_and_rejects_completed_or_answered(self):
        for mode in ['valid','complete','answered']:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                parent=Path(d)/'source';out=Path(d)/'retry'
                write_json(parent/'memory/V1/memory.json',{'complete':mode=='complete','completed_windows':['W1']})
                if mode=='answered':write_json(parent/'videos/V1/graph/predictions.jsonl',{'status':'ok'})
                before=(parent/'memory/V1/memory.json').read_bytes()
                if mode!='valid':
                    with self.assertRaises(AssertionError):clone_checkpoint(parent,out,'V1')
                else:
                    clone_checkpoint(parent,out,'V1')
                    self.assertEqual(before,(out/'memory/V1/memory.json').read_bytes())
                    (out/'memory/V1/memory.json').write_text('changed retry')
                    clone_checkpoint(parent,out,'V1')
                    self.assertEqual((out/'memory/V1/memory.json').read_text(),'changed retry')
                self.assertEqual(before,(parent/'memory/V1/memory.json').read_bytes())


if __name__=='__main__':unittest.main()
