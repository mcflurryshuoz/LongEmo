"""Anthropic SSE transport; preserves normal parsing and refusal handling."""
import json
import time
from urllib import request, error

from evaluation.clients import Client, ServiceError
from evaluation.inference.adapters import anthropic
from experiments.zyf.alternative_model_probe import claude_headers
from experiments.zyf.mirror_stream_probe import events, collect
from methods.longemo.common import append_json, fingerprint


class StreamClient(Client):
    ledger = None

    def generate(self, messages):
        assert self.api_format == 'anthropic' and self.options == {'stream': True}
        stats = {'started_monotonic': time.monotonic(), 'request_hash': fingerprint(messages),
                 'time_unix': time.time(), 'http_status': None, 'message_stop_seen': False}
        try:
            req = request.Request(self._url(), data=json.dumps(self.payload(messages), allow_nan=False).encode(),
                                  headers=claude_headers(self), method='POST')
            try:
                with request.urlopen(req, timeout=self.timeout) as response:
                    stats.update(http_status=response.status, headers_seconds=round(time.monotonic()-stats['started_monotonic'], 3),
                                 content_type=response.headers.get_content_type())
                    if stats['content_type'] != 'text/event-stream':
                        raise ValueError('provider did not return an SSE stream')
                    def bounded_lines():
                        for line in response:
                            if time.monotonic()-stats['started_monotonic'] > self.timeout:
                                raise TimeoutError('stream total duration exceeded configured timeout')
                            yield line
                    raw = collect(events(bounded_lines()), stats)
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
            result = anthropic.parse_response(self, raw)
            stats['status'] = 'ok'
            return result
        except Exception as exc:
            stats.update(status='error', error_type=type(exc).__name__)
            if isinstance(exc, ServiceError):
                stats.update(http_status=exc.status_code, service_error_code=exc.code)
            raise
        finally:
            stats['elapsed_seconds'] = round(time.monotonic()-stats.pop('started_monotonic'), 3)
            if self.ledger:
                append_json(self.ledger, stats)
