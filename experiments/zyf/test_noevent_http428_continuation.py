import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.zyf import noevent_http428_continuation as c
from experiments.zyf import test_noevent_probe_seed as fixtures
from experiments.zyf import noevent_probe_seed as seed, noevent_retry_probe as probe
from evaluation.clients import Client

class Tests(unittest.TestCase):
    def test_exact_source_route_imports_audio_and_vision_without_api(self):
        for stage in ('audio','visual'):
            with self.subTest(stage=stage),tempfile.TemporaryDirectory() as tmp:
                parent,run,repo,source,target,diagnostic,prepared,checkpoint=fixtures.Tests().fixture(Path(tmp),stage)
                before={str(p):probe.sha(p) for p in parent.rglob('*') if p.is_file()}
                original=probe.prepare
                with c.reconstructed_source(prepared,parent,repo,'V'),patch.object(Client,'generate',side_effect=AssertionError('network forbidden')):
                    receipt=seed.apply_seed(parent,run,'V',diagnostic,repo)
                self.assertIs(probe.prepare,original)
                self.assertEqual(before,{str(p):probe.sha(p) for p in parent.rglob('*') if p.is_file()})
                self.assertEqual(receipt['stage'],stage)
                memory=json.loads((target/'memory.json').read_text())
                self.assertEqual(len(memory['completed_windows']),1 if stage=='audio' else 2)
                self.assertEqual(seed.verify_seed_receipt(run,'V',checkpoint),receipt)

    def test_source_route_fails_closed_and_restores_after_error(self):
        original=probe.prepare
        with self.assertRaises(ValueError):
            with c.reconstructed_source({},Path('/parent'),Path('/core'),'V'):
                probe.prepare('/wrong','V','/core')
        self.assertIs(probe.prepare,original)

    def test_fixed_selection_refuses_unfinished_or_changed_successes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);(tmp/'spec.json').write_text('{}')
            for state in ({'status':'running','results':{}},{'status':'finished','results':{v:{'classification':'explicit_dlp_rejection'} for v in c.diagnosis.CASES}}):
                (tmp/'finished.json').write_text(json.dumps(state))
                with self.assertRaises(ValueError):c.successful_cases(tmp/'spec.json',tmp)

    def test_credential_comes_from_frozen_task_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();credential=root/'private.json';credential.write_text('{}')
            task=root/'task.json';task.write_text(json.dumps({'command':['python','--credential-file',str(credential)]}))
            self.assertEqual(c.original_credential({'V':{'task_path':str(task)}}),str(credential))
            task.write_text(json.dumps({'command':[]}))
            with self.assertRaises(ValueError):c.original_credential({'V':{'task_path':str(task)}})

    def test_resume_cannot_delete_or_reset_any_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();run=root/'run';run.mkdir();selection=run/'selection.json';selection.write_text('{}')
            audit=root/'audit.json';audit.write_text(json.dumps({'run':str(run),'reason':'local_prepare_KeyError_credential_file_before_any_API','run_files':{str(selection):probe.sha(selection)},'claims':{}}))
            c.empty_preparation(run,audit)
            (run/'task.json').write_text('{}')
            with self.assertRaises(ValueError):c.empty_preparation(run,audit)
