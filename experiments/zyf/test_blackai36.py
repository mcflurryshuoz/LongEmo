import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from experiments.zyf.blackai36_continuation import clone_checkpoint, score_guard
from experiments.zyf.blackai36_worker import visual_client,audio_client,MODEL
from experiments.zyf.blackai_continuation import BLACKAI
from experiments.zyf.recover_blackai import transient_failure
from methods.longemo.common import file_hash

class Tests(unittest.TestCase):
 def test_clone_archives_pending_audio_preserves_completed_and_source(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);src=root/'source';(src/'audio').mkdir(parents=True)
   (src/'memory.json').write_text('{"completed_windows":["W00001"]}')
   (src/'manifest.json').write_text('{}')
   (src/'calls.jsonl').write_text('{"status":"error","http_status":402}\n')
   (src/'audio/W00001.json').write_text('{"old":"complete"}')
   (src/'audio/W00002.json').write_text('{"old":"pending"}')
   before={str(p.relative_to(src)):file_hash(p) for p in src.rglob('*') if p.is_file()}
   dst=root/'target';r=clone_checkpoint(src,dst,'parent')
   self.assertEqual(before,r['files']);self.assertEqual(before,{str(p.relative_to(src)):file_hash(p) for p in src.rglob('*') if p.is_file()})
   self.assertTrue((dst/'audio/W00001.json').exists());self.assertFalse((dst/'audio/W00002.json').exists())
   self.assertTrue((dst/'ancestry/parent/pending_audio/W00002.json').exists());self.assertFalse((dst/'calls.jsonl').exists())
   self.assertEqual(r,clone_checkpoint(src,dst,'parent'))
 def test_complete_source_refused(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);src=root/'source';src.mkdir();(src/'memory.json').write_text('{"complete":true,"completed_windows":[]}')
   with self.assertRaises(AssertionError):clone_checkpoint(src,root/'target','p')
 def test_unknown_existing_checkpoint_refused(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   with self.assertRaises(FileNotFoundError):clone_checkpoint(p,p,'p')
 def test_exact_validated_profiles_and_secret_not_in_config(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'secret.json';p.write_text('{"GEMINI_API_KEY":"test-only-secret"}')
   a=SimpleNamespace(model=MODEL,base_url=BLACKAI,max_tokens=8192,thinking='default',credential_file=p,with_audio=True,audio_model=MODEL,audio_base_url=BLACKAI)
   for c in (visual_client(a),audio_client(a)):
    self.assertEqual(c.options,{});self.assertIsNone(c.temperature);self.assertNotIn('test-only-secret',json.dumps(c.configuration()))
   a.with_audio=False;self.assertIsNone(audio_client(a))
 def test_transient_only_recovery(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);folder=root/'memory/V';folder.mkdir(parents=True)
   p=folder/'calls.jsonl'
   for status,code,want in [(524,None,True),(520,None,True),(402,'BALANCE_INSUFFICIENT',False),(200,'content_filter',False),(None,None,False)]:
    p.write_text(json.dumps({'purpose':'W1','status':'error','http_status':status,'service_error_code':code})+'\n')
    self.assertEqual(transient_failure(root,'V'),want)
 def test_score_guard_rejects_changed_hash_and_overlap(self):
  with tempfile.TemporaryDirectory() as d:
   rt=Path(d);p=rt/'runs/old/scores.jsonl';p.parent.mkdir(parents=True)
   p.write_text(''.join(json.dumps({'question_id':f'Q{i}','status':'ok'})+'\n' for i in range(486)))
   hashes={'old':file_hash(p)}
   with patch('experiments.zyf.blackai36_continuation.SCORE_HASHES',hashes):
    self.assertEqual(score_guard(rt,[{'question_id':'NEW'}]),486)
    with self.assertRaises(AssertionError):score_guard(rt,[{'question_id':'Q0'}])
    p.write_text(p.read_text()+'\n')
    with self.assertRaises(AssertionError):score_guard(rt,[{'question_id':'NEW'}])
if __name__=='__main__':unittest.main()
