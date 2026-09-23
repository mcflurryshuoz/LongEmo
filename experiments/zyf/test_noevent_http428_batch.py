import json
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch
from experiments.zyf import noevent_http428_batch as b

class Tests(TestCase):
    def test_service_errors_stop_next_wave_but_per_case_dlp_does_not(self):
        for code in (401,402,403,429):self.assertTrue(b.stop_service({'http_status':code}))
        for status in ('transport_error','invalid_response_json'):self.assertTrue(b.stop_service({'status':status}))
        self.assertFalse(b.stop_service({'http_status':428,'classification':'explicit_dlp_rejection'}))

    def test_auth_failure_finishes_inflight_wave_without_starting_later_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);spec=tmp/'spec.json';spec.write_text(json.dumps({'cases':[{'video_id':str(i),'core_repo':'/frozen','question_ids':[str(i)]} for i in range(9)]}))
            calls=[]
            def popen(args,**kwargs):
                vid=args[args.index('--video-id')+1];calls.append(vid)
                out=Path(args[args.index('--output-dir')+1]);out.mkdir()
                (out/'diagnosis.json').write_text(json.dumps({'status':'http_error','http_status':401 if vid=='0' else 428}))
                class P:
                    pid=123
                    def wait(self):return 0
                return P()
            with patch.object(b.diag,'load_case'),patch.object(b,'ticks',return_value='1'),patch.object(b.subprocess,'Popen',side_effect=popen):
                result=b.batch('probe',spec,tmp/'output')
            self.assertEqual(set(calls),{'0','1','2','3'})
            self.assertEqual(result['status'],'stopped_for_service_or_audit')
            self.assertEqual(len(result['results']),4)
            with patch.object(b.diag,'load_case'),patch.object(b.subprocess,'Popen') as p:
                with self.assertRaises(ValueError):b.batch('probe',spec,tmp/'output')
                p.assert_not_called()
