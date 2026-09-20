import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.clients import Client
from experiments.zyf.aicodemirror_probe import BASE, mirror_headers
from experiments.zyf.aicodemirror_worker import verified_inherited_audio
from methods.longemo.audio import AUDIO_PROMPT
from methods.longemo.common import fingerprint


class MirrorTests(unittest.TestCase):
    def test_bearer_auth_is_provider_scoped(self):
        c=Client('gemini-3.7-flash',BASE,'gemini','secret')
        self.assertEqual(mirror_headers(c)['Authorization'],'Bearer secret')
        c.base_url='https://www.blackaicoding.com/v1beta'
        self.assertNotIn('Authorization',mirror_headers(c))
        self.assertEqual(mirror_headers(c)['x-goog-api-key'],'secret')

    def test_inherited_audio_has_exact_input_and_no_old_provider_request(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'audio.json'
            audio={'type':'input_audio','input_audio':{'format':'wav','data':'AA=='}}
            old=Client('gemini-3.7-flash','https://www.blackaicoding.com/v1beta','gemini')
            new=Client('gemini-3.7-flash',BASE,'gemini')
            messages=[{'role':'system','content':AUDIO_PROMPT},{'role':'user','content':[
                {'type':'text','text':json.dumps({'clip_start':0,'clip_end':1})},audio]}]
            cached={'model':old.configuration(),'input_fingerprint':fingerprint({'messages':messages,'model':old.configuration()}),
                    'result':{'observations':[{'span':[0,1],'voice':'speaker','cue':'hello','uncertainty':''}]}}
            p.write_text(json.dumps(cached))
            with patch.object(Client,'generate',side_effect=AssertionError('must not call API')):
                parts,trace=verified_inherited_audio([audio],[0,1],new,p)
            self.assertTrue(trace['inherited']);self.assertEqual(trace['base_url'],old.base_url)
            self.assertEqual(parts[0]['type'],'text')
            with self.assertRaises(AssertionError):verified_inherited_audio([audio],[0,2],new,p)


if __name__=='__main__':unittest.main()
