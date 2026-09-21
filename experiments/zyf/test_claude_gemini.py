import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from evaluation.clients import Client, ServiceError
from experiments.zyf.claude_gemini_worker import audio_client, visual_client, refusal_aware_parser
from experiments.zyf.claude_gemini_continuation import clone_memory, canary_passed, PARENT
from methods.longemo.common import file_hash


class HybridTests(unittest.TestCase):
    def test_explicit_clients_match_validated_profiles(self):
        with tempfile.TemporaryDirectory() as d:
            secret = Path(d)/'private.json';secret.write_text('{"GEMINI_API_KEY":"test-secret"}')
            args = SimpleNamespace(model='claude-opus-5',base_url='https://api.aicodemirror.ai/api/claudecode/v1',max_tokens=8192,
                thinking='default',credential_file=secret,timeout=240,audio_model='gemini-2.5-pro',audio_base_url='https://api.aicodemirror.ai/api/gemini/v1beta')
            visual = visual_client(args);audio = audio_client(args)
            self.assertEqual(visual.api_format,'anthropic');self.assertIsNone(visual.temperature)
            self.assertEqual(audio.options,{'generationConfig':{'thinkingConfig':{'thinkingBudget':1024}}})
            self.assertNotIn('test-secret',json.dumps(visual.configuration()))

    def test_refusal_is_not_retried_as_invalid_json(self):
        parse = refusal_aware_parser(lambda client,raw: raw)
        with self.assertRaises(ServiceError) as raised:parse(None,{'stop_reason':'refusal'})
        self.assertEqual(raised.exception.code,'content_filter');self.assertFalse(raised.exception.retryable)
        self.assertEqual(parse(None,{'stop_reason':'end_turn'}),{'stop_reason':'end_turn'})

    def test_clone_keeps_audio_and_archives_old_failures_without_changing_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.mkdir();(source/'audio').mkdir()
            (source/'memory.json').write_text('{"completed_windows":["W00001"]}')
            (source/'manifest.json').write_text('{}');(source/'calls.jsonl').write_text('{"status":"error"}\n')
            (source/'audio/W00002.json').write_text('{"source":"old_audio"}')
            before={str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()}
            target=root/'target';receipt=clone_memory(source,target)
            self.assertEqual(before,receipt['files'])
            self.assertEqual(before,{str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()})
            self.assertTrue((target/'audio/W00002.json').exists())
            self.assertFalse((target/'calls.jsonl').exists())
            self.assertTrue((target/'ancestry'/PARENT/'calls.jsonl').exists())
            self.assertEqual(receipt,clone_memory(source,target))

    def test_first_window_failure_and_unknown_destination(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.mkdir();(source/'manifest.json').write_text('{}')
            self.assertEqual(clone_memory(source,root/'target')['inherited_windows'],[])
            unknown=root/'unknown';unknown.mkdir()
            with self.assertRaises(FileNotFoundError):clone_memory(source,unknown)

    def test_gate_requires_a_complete_canary_memory(self):
        self.assertFalse(canary_passed({'videos':{'G2_V000137':{'status':'frontend_failed'}}}))
        self.assertTrue(canary_passed({'videos':{'G2_V000137':{'status':'memory_complete'}}}))


if __name__ == '__main__':unittest.main()
