import hashlib,json,tempfile
from pathlib import Path
import unittest
from experiments.zyf import noevent_final_audit as a
class Tests(unittest.TestCase):
    def test_dlp_requires_exact_body_hash_and_http_status(self):
        body=json.dumps({'error':{'message':'Data leak protection rejected'}});r={'http_status':428,'body_text':body,'body_sha256':hashlib.sha256(body.encode()).hexdigest()}
        self.assertTrue(a.exact_dlp(r));self.assertFalse(a.exact_dlp({**r,'http_status':400}));self.assertFalse(a.exact_dlp({**r,'body_sha256':'wrong'}));self.assertFalse(a.exact_dlp({**r,'body_redacted':True}))
    def test_first_window_refusal_does_not_fallback_to_pilot(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t).resolve();f=root/'noevent/videos/V/memory/V';f.mkdir(parents=True);(f/'manifest.json').write_text('{}');(f/'audio').mkdir();(f/'audio/calls.jsonl').write_text(json.dumps({'purpose':'audio:V:W00001','status':'error','time_unix':1,'service_error_code':'content_filter'})+'\n')
            self.assertEqual(a.memory_folder(root,'V'),f);row,ledger,w,n=a.final_failure(f);self.assertEqual((w,n),('W00001',0));self.assertEqual(ledger.parent.name,'audio')
    def test_only_next_window_failure_can_be_final(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t).resolve();(p/'memory.json').write_text(json.dumps({'completed_windows':['W00001']}));data=[{'purpose':'V:W00001','status':'error','time_unix':9},{'purpose':'V:W00002','status':'error','time_unix':2}];(p/'calls.jsonl').write_text('\n'.join(map(json.dumps,data)))
            row,_,w,_=a.final_failure(p);self.assertEqual(row['time_unix'],2);self.assertEqual(w,'W00002')
    def test_grouped_missing_excluded_real_zero_retained(self):
        q=[{'question_id':x,'type':'T'} for x in ('a','b','c')];r=a.grouped(q,{'a':{'normalized_score':0},'b':{'normalized_score':1}},'type');self.assertEqual(r,{'T':{'scored':2,'total':3,'mean_scored':50}})
    def test_series_does_not_guess_url_source(self):
        self.assertEqual(a.series('jiayouernv S1E4'),'Home with Kids');self.assertEqual(a.series('https://example.org/video'),'unclassified')
