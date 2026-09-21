import json
import time
import unittest

from evaluation.clients import ServiceError
from experiments.zyf.mirror_stream_probe import events, collect


def response():
    return [
        {'type': 'message_start', 'message': {'content': [], 'model': 'claude-opus-5', 'usage': {'input_tokens': 7}}},
        {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
        {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': '{"a":'}},
        {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': '1}'}},
        {'type': 'content_block_stop', 'index': 0},
        {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 5}},
        {'type': 'message_stop'},
    ]


class StreamTests(unittest.TestCase):
    def run_events(self, rows):
        stats = {'started_monotonic': time.monotonic()}
        wire = [line for row in rows for line in [b'event: test\n', ('data: '+json.dumps(row)+'\n').encode(), b'\n']]
        return collect(events(wire), stats), stats

    def test_complete_text_and_cumulative_usage(self):
        raw, stats = self.run_events(response())
        self.assertEqual(raw['content'], [{'type':'text','text':'{"a":1}'}])
        self.assertEqual(raw['usage'], {'input_tokens':7,'output_tokens':5})
        self.assertTrue(stats['message_stop_seen'])

    def test_truncated_generation_never_accepted(self):
        with self.assertRaisesRegex(ValueError, 'without message_stop'):
            self.run_events(response()[:-1])

    def test_in_stream_error_is_not_success(self):
        rows = response()[:3]+[{'type':'error','error':{'type':'overloaded_error'}}]
        with self.assertRaises(ServiceError) as ctx:
            self.run_events(rows)
        self.assertEqual(ctx.exception.status_code, 529)

    def test_refusal_is_preserved_for_client_check(self):
        rows = response();rows[-2]['delta']['stop_reason']='refusal'
        self.assertEqual(self.run_events(rows)[0]['stop_reason'], 'refusal')

    def test_delta_after_block_stop_rejected(self):
        rows=response();rows.insert(-2, rows[2])
        with self.assertRaisesRegex(ValueError, 'without an open block'):
            self.run_events(rows)

    def test_incomplete_sse_event_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unterminated SSE'):
            list(events([b'data: {"type":"ping"}\n']))


if __name__ == '__main__':
    unittest.main()
