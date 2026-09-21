"""One transport-only diagnostic on an immutable HTTP-524 failed window.

Protocol: https://platform.claude.com/docs/en/build-with-claude/streaming
No schema relaxation, output repair, automatic retry, or benchmark mutation.
"""
import argparse
import fcntl
import json
import signal
import time
from pathlib import Path
from urllib import request, error

from evaluation.clients import Client, ServiceError
from evaluation.inference.adapters import anthropic
from experiments.zyf import aicodemirror_probe as probe
from experiments.zyf.alternative_model_probe import CLAUDE_BASE, claude_headers
from experiments.zyf.blackai_continuation import atomic, read
from methods.longemo.common import file_hash, fingerprint
from methods.longemo.media import window_input, subtitles
from methods.longemo.memory import apply_window
from methods.longemo.prompts import PERCEPTION
from evaluation.inference.prompts import json_object

PARENT = 'aicodemirror_recharge_parallel_20260921'
VIDEO = 'G2_V000104'
EXPECTED_REQUEST = 'c63ffa057013236f066f83378ec38615308e3edf34554490463bc2f5e73befe1'


def events(lines):
    data = []
    size = 0
    for line in lines:
        size += len(line)
        if size > 8_000_000:
            raise ValueError('stream exceeds diagnostic byte limit')
        line = line.decode('utf-8').rstrip('\r\n')
        if line == '':
            if data:
                event = json.loads('\n'.join(data))
                if not isinstance(event, dict):
                    raise ValueError('stream event must be an object')
                yield event
                data = []
        elif line.startswith('data:'):
            value = line[5:]
            data.append(value[1:] if value.startswith(' ') else value)
    if data:
        raise ValueError('unterminated SSE event')


def collect(event_source, stats):
    raw = None
    blocks = {}
    open_blocks = set()
    for event in event_source:
        kind = event.get('type')
        stats['event_count'] = stats.get('event_count', 0) + 1
        if kind == 'error':
            code = (event.get('error') or {}).get('type')
            raise ServiceError(529 if code == 'overloaded_error' else 200, code)
        if kind == 'message_start':
            if raw is not None:
                raise ValueError('duplicate message start')
            raw = dict(event['message'])
            if raw.get('content'):
                raise ValueError('nonempty initial streaming content')
        elif kind == 'content_block_start':
            index = event['index']
            if raw is None or index in blocks:
                raise ValueError('invalid block start')
            block = dict(event['content_block'])
            if block.get('type') not in {'text', 'thinking', 'redacted_thinking'}:
                raise ValueError('unsupported diagnostic block type')
            blocks[index] = block if block['type'] == 'text' else {'type': block['type']}
            open_blocks.add(index)
        elif kind == 'content_block_delta':
            index = event['index']
            if index not in open_blocks:
                raise ValueError('delta without an open block')
            delta = event['delta']
            if delta.get('type') == 'text_delta':
                if blocks[index]['type'] != 'text':
                    raise ValueError('text delta in another block type')
                blocks[index]['text'] = blocks[index].get('text', '') + delta['text']
                stats.setdefault('first_text_seconds', round(time.monotonic() - stats['started_monotonic'], 3))
            elif blocks[index]['type'] != 'thinking' or delta.get('type') not in {'thinking_delta', 'signature_delta'}:
                raise ValueError('unsupported diagnostic delta')
        elif kind == 'content_block_stop':
            if event['index'] not in open_blocks:
                raise ValueError('stop without an open block')
            open_blocks.remove(event['index'])
        elif kind == 'message_delta':
            if raw is None:
                raise ValueError('message delta before start')
            raw.update(event.get('delta') or {})
            raw.setdefault('usage', {}).update(event.get('usage') or {})
        elif kind == 'message_stop':
            if raw is None or open_blocks:
                raise ValueError('message stop before completed blocks')
            raw['content'] = [blocks[i] for i in sorted(blocks) if blocks[i]['type'] == 'text']
            stats['message_stop_seen'] = True
            return raw
    raise ValueError('stream ended without message_stop')


class StreamClient(Client):
    audit_folder = None

    def generate(self, messages):
        assert self.api_format == 'anthropic' and self.model == 'claude-opus-5'
        assert self.base_url == CLAUDE_BASE and self.timeout == 240 and self.max_tokens == 8192
        assert fingerprint(messages) == EXPECTED_REQUEST, 'reconstructed failed request differs; no API call'
        record = self.audit_folder/'request_started.json'
        assert not record.exists(), 'diagnostic already attempted; no repeat'
        payload = self.payload(messages)
        assert 'stream' not in payload
        atomic(record, {'request_hash': fingerprint(messages), 'nonstream_payload_hash': fingerprint(payload),
                       'time_unix': time.time(), 'changed_payload_fields': ['stream'], 'attempt_limit': 1})
        payload['stream'] = True
        stats = {'started_monotonic': time.monotonic(), 'http_status': None, 'message_stop_seen': False}
        def deadline(signum, frame):
            raise TimeoutError('original 240-second total request limit exceeded')
        previous = signal.signal(signal.SIGALRM, deadline)
        signal.setitimer(signal.ITIMER_REAL, self.timeout)
        try:
            req = request.Request(self._url(), data=json.dumps(payload, allow_nan=False).encode(),
                                  headers=claude_headers(self), method='POST')
            try:
                with request.urlopen(req, timeout=self.timeout) as response:
                    stats.update(http_status=response.status, headers_seconds=round(time.monotonic()-stats['started_monotonic'], 3),
                                 content_type=response.headers.get_content_type())
                    if stats['content_type'] != 'text/event-stream':
                        raise ValueError('provider did not return an SSE stream')
                    raw = collect(events(response), stats)
            except error.HTTPError as exc:
                stats['http_status'] = exc.code
                try:
                    detail = json.load(exc).get('error', {})
                    code = detail.get('code', detail.get('type')) if isinstance(detail, dict) else None
                except (ValueError, AttributeError):
                    code = None
                raise ServiceError(exc.code, code) from None
            if raw.get('stop_reason') == 'refusal':
                raise ServiceError(200, 'content_filter')
            return anthropic.parse_response(self, raw)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
            stats['elapsed_seconds'] = round(time.monotonic()-stats.pop('started_monotonic'), 3)
            atomic(self.audit_folder/'transport.json', stats)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--credential-file', type=Path, required=True)
    a = p.parse_args()
    run = a.runtime/'runs'/PARENT
    state = read(run/'frontend_status.json')
    assert VIDEO not in state.get('active_videos', [])
    assert state['videos'][VIDEO]['status'] == 'frontend_failed'
    assert state['videos'][VIDEO]['attempts'] == 4
    worker = read(run/'videos'/VIDEO/'worker_process.json')
    assert not Path('/proc', str(worker['pid'])).exists()
    memory = run/'memory'/VIDEO
    before = {str(f.relative_to(memory)): file_hash(f) for f in memory.rglob('*') if f.is_file()}
    last = json.loads((memory/'calls.jsonl').read_text().splitlines()[-1])
    assert last['http_status'] == 524 and last['request_hash'] == EXPECTED_REQUEST
    directory = a.runtime/'diagnostics/mirror_stream_v104_20260921'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'probe.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        StreamClient.audit_folder = directory
        assert not (directory/'request_started.json').exists()
        atomic(directory/'protocol.json', {'parent_run': PARENT, 'video_id': VIDEO, 'window_id': 'W00058',
            'attempt_limit': 1, 'expected_request_hash': EXPECTED_REQUEST, 'parent_file_hashes': before,
            'code_sha256': file_hash(Path(__file__)), 'scope': 'Transport-only single-window diagnostic; never a benchmark score.',
            'model_media_prompt_schema_safety_unchanged': True})
        result = {'video_id': VIDEO, 'window_id': 'W00058', 'attempt_limit': 1,
                  'scope': 'Transport-only fixed failed visual window, no benchmark scores', 'time_unix': time.time()}
        started = time.monotonic()
        try:
            graph = read(memory/'memory.json')
            core = [1140.0, 1160.0]; interval = [1138.0, 1162.0]
            media, metadata = window_input(a.runtime/'data/episode/videos'/(VIDEO+'.mp4'), *interval,
                fps=1, max_frames=16, max_pixels=150528, with_audio=True,
                subtitle_rows=subtitles(a.runtime/'data/prepared_subtitles'/(VIDEO+'.json')))
            cached = memory/'audio/W00058.json'; audio = read(cached)
            media = [part for part in media if part['type'] != 'input_audio'] + [
                {'type': 'text', 'text': 'Timestamped audio observations from an independent audio model; '
                 'these may contain errors. Match voice identity cautiously using the frames and dialogue. ' + json.dumps(audio['result'], ensure_ascii=False)}]
            metadata.update(audio_representation='derived_timestamped_cues',
                audio_observer={'model': audio['model']['model'], 'input_fingerprint': audio['input_fingerprint'], 'source_sha256': file_hash(cached)})
            context = {'video_id': VIDEO, 'window_id': 'W00058', 'core_interval': core, 'media_interval': interval,
                       'corrections_enabled': False, 'cast': graph['entities'], 'preceding_events': graph['events'][-3:]}
            messages = [{'role': 'system', 'content': PERCEPTION}, {'role': 'user', 'content':
                [{'type': 'text', 'text': json.dumps(context, ensure_ascii=False)}] + media}]
            result['request_hash'] = fingerprint(messages)
            client = StreamClient('claude-opus-5', CLAUDE_BASE, 'anthropic', read(a.credential_file)['GEMINI_API_KEY'], 240, 8192, None, {})
            response = client.generate(messages)
            payload = json_object(response['content'])
            apply_window(graph, payload, window_id='W00058', core=core, media=interval, metadata=metadata, allow_revisions=False)
            atomic(directory/'validated_payload.json', payload)
            result.update(status='ok', finish_reason=response['finish_reason'], usage=response['usage'],
                          response_model=response['raw_response'].get('model'))
        except Exception as exc:
            result.update(status='error', error_type=type(exc).__name__)
            if isinstance(exc, ServiceError):
                result.update(http_status=exc.status_code, service_error_code=exc.code)
            elif isinstance(exc, (AssertionError, ValueError, TimeoutError)):
                result['validation_error'] = str(exc)[:300]
        result['elapsed_seconds'] = round(time.monotonic()-started, 3)
        atomic(directory/'result.json', result)
        after = {str(f.relative_to(memory)): file_hash(f) for f in memory.rglob('*') if f.is_file()}
        assert before == after, 'diagnostic must not mutate original memory'
        atomic(directory/'audit.json', {'parent_files_unchanged': True, 'benchmark_mutated': False})
        print(json.dumps({k:v for k,v in result.items() if k not in {'usage','client_configuration'}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
