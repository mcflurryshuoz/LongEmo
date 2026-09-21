import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.zyf import blackai31_worker as worker
from experiments.zyf.remaining47_continuation import clone_checkpoint, guard, SCORE_HASHES
from methods.longemo import runner
from methods.longemo.common import file_hash
from methods.longemo.memory import empty_memory


class Tests(unittest.TestCase):
    def test_clone_keeps_successful_windows_archives_foreign_pending_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';(source/'audio').mkdir(parents=True)
            (source/'memory.json').write_text(json.dumps({'complete':False,'completed_windows':['W00001']}))
            (source/'audio/W00001.json').write_text('{"old_valid":true}')
            (source/'audio/W00002.json').write_text('{"foreign_pending":true}')
            (source/'manifest.json').write_text('{}');(source/'calls.jsonl').write_text('old ledger\n')
            before={str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()}
            target=root/'target';clone_checkpoint(source,target,'parent')
            self.assertEqual(before,{str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()})
            self.assertEqual((source/'audio/W00001.json').read_bytes(),(target/'audio/W00001.json').read_bytes())
            self.assertFalse((target/'audio/W00002.json').exists())
            self.assertTrue((target/'ancestry/parent/pending_audio/W00002.json').exists())

    def fixture(self,root):
        (root/'videos').mkdir();(root/'videos/V.mp4').write_bytes(b'video fixture')
        (root/'subtitles').mkdir();(root/'subtitles/V.json').write_text('[]')
        folder=root/'memory/V';folder.mkdir(parents=True)
        (folder/'memory.json').write_text(json.dumps(empty_memory('V',40,file_hash(root/'videos/V.mp4'))))
        (root/'credential.json').write_text('{"GEMINI_API_KEY":"test-key"}')
        return SimpleNamespace(output_dir=root/'memory',videos_dir=root/'videos',subtitles_dir=root/'subtitles',
            credential_file=root/'credential.json',model=worker.MODEL,base_url=worker.BLACKAI,
            audio_model=worker.MODEL,audio_base_url=worker.BLACKAI,max_tokens=8192,thinking='default',
            with_audio=True,window_seconds=20.0,padding=2.0,fps=1,max_frames=16,max_pixels=150528,
            allow_revisions=False,tries=3),folder

    def media(self,*args,**kwargs):
        return [{'type':'input_audio','input_audio':{'format':'wav','data':'AAAA'}}],{'audio':True,'subtitle_ids':[]}

    def response(self,payload):
        return {'content':json.dumps(payload),'usage':{},'raw_response':{'modelVersion':'test-response-model'}}

    def test_gate_window_reused_by_full_builder_without_api_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,folder=self.fixture(Path(tmp))
            audio=self.response({'observations':[]});visual=self.response({k:[] for k in ['entities','observations','events','relations','corrections']})
            with patch.object(runner,'probe',return_value={'duration':40}), patch.object(runner,'window_input',side_effect=self.media), patch.object(runner,'audio_client_for',worker.audio_client), patch.object(worker.Client,'generate',side_effect=[audio,visual,audio,visual]) as generate:
                probe=worker.probe_one('V',args,worker.visual_client(args))
                saved=(folder/'windows/W00001.json').read_bytes()
                self.assertFalse(probe['complete']);self.assertEqual(generate.call_count,2)
                runner._build_video('V',args,worker.visual_client(args))
                self.assertEqual(generate.call_count,4)
                self.assertEqual(saved,(folder/'windows/W00001.json').read_bytes())
                graph=json.loads((folder/'memory.json').read_text())
                self.assertTrue(graph['complete']);self.assertEqual(graph['completed_windows'],['W00001','W00002'])

    def test_failed_gate_does_not_commit_or_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,folder=self.fixture(Path(tmp));before=(folder/'memory.json').read_bytes()
            with patch.object(runner,'probe',return_value={'duration':40}), patch.object(runner,'window_input',side_effect=self.media), patch.object(worker.Client,'generate',return_value=self.response({'observations':[{'span':[0,1],'voice':'','cue':'x'}]})) as generate:
                with self.assertRaises(RuntimeError):worker.probe_one('V',args,worker.visual_client(args))
                with self.assertRaises(AssertionError):worker.probe_one('V',args,worker.visual_client(args))
                self.assertEqual(generate.call_count,1);self.assertEqual(before,(folder/'memory.json').read_bytes())
                self.assertFalse((folder/'probe_result.json').exists())

    def test_guard_rejects_overlap_with_any_of_511_first_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);names=list(SCORE_HASHES)
            for i,name in enumerate(names):
                p=root/'runs'/name/'scores.jsonl';p.parent.mkdir(parents=True)
                rows=[{'question_id':'Q%d'%j,'status':'ok'} for j in (range(511) if i==0 else [])]
                p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
            with patch('experiments.zyf.remaining47_continuation.file_hash',side_effect=lambda p:SCORE_HASHES[p.parent.name]):
                with self.assertRaises(AssertionError):guard(root,[{'question_id':'Q510'}])
                guard(root,[{'question_id':'new'}])


class ProfileTests(unittest.TestCase):
    def test_profile_and_gate_use_correct_distinct_clients(self):
        from experiments.zyf import remaining47_worker as profile
        with tempfile.TemporaryDirectory() as tmp:
            args,folder=Tests().fixture(Path(tmp))
            args.model=profile.MODEL;args.base_url=profile.VISUAL_BASE
            args.audio_model=profile.AUDIO_MODEL;args.audio_base_url=profile.AUDIO_BASE
            self.assertEqual(profile.visual_client(args).options, {'stream':True})
            self.assertEqual(profile.audio_client(args).model, 'gemini-2.5-pro')
            response=Tests().response
            payload={k:[] for k in ['entities','observations','events','relations','corrections']}
            with patch.object(runner,'probe',return_value={'duration':40}), patch.object(runner,'window_input',side_effect=Tests().media), patch.object(worker,'audio_client',profile.audio_client), patch.object(runner,'audio_client_for',profile.audio_client), patch.object(profile.Client,'generate',return_value=response({'observations':[]})) as audio, patch.object(profile.StreamClient,'generate',return_value=response(payload)) as visual:
                worker.probe_one('V',args,profile.visual_client(args))
                saved=(folder/'windows/W00001.json').read_bytes()
                runner._build_video('V',args,profile.visual_client(args))
                self.assertEqual((audio.call_count,visual.call_count),(2,2))
                self.assertEqual(saved,(folder/'windows/W00001.json').read_bytes())
                self.assertTrue(json.loads((folder/'memory.json').read_text())['complete'])


if __name__=='__main__':unittest.main()
