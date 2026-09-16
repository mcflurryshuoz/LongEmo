"""Azure CLI authentication and cross-process throttling for Chat Completions.

Tokens remain in process memory. A SQLite reservation ledger coordinates only
this experiment's calls; Retry-After also pauses all its workers on shared 429s.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import threading
import time
from urllib import request, error, parse
import uuid

_TOKEN_LOCK = threading.Lock()
_TOKENS = {}


def is_azure(base_url):
    p = parse.urlparse(base_url)
    return p.scheme == 'https' and bool(p.hostname) and p.hostname.endswith(('.cognitiveservices.azure.com', '.openai.azure.com'))


def configuration():
    return {'provider': 'azure', 'api_version': os.getenv('AZURE_OPENAI_API_VERSION', '2025-01-01-preview'),
            'authentication': 'azure_cli', 'credential_context': os.getenv('AZURE_CONFIG_DIR', str(Path.home()/'.azure'))}


def access_token(*, refresh=False):
    context = configuration()['credential_context']
    with _TOKEN_LOCK:
        cached = _TOKENS.get(context)
        if cached and cached[1] > time.time()+300 and not refresh:
            return cached[0]
        env = os.environ.copy()
        env['AZURE_CONFIG_DIR'] = context
        result = subprocess.run(['az', 'account', 'get-access-token', '--resource',
            'https://cognitiveservices.azure.com/', '--output', 'json'], env=env,
            capture_output=True, text=True, timeout=40)
        if result.returncode:
            raise AzureError(401, 'AzureCliLoginRequired', retryable=False)
        data = json.loads(result.stdout)
        token = data['accessToken']
        # Azure CLI returns epoch expires_on; conservative fallback avoids
        # depending on the locale/time zone of its legacy expiresOn string.
        expiry = float(data.get('expires_on', time.time()+300))
        _TOKENS[context] = (token, expiry)
        return token


class AzureError(RuntimeError):
    def __init__(self, status, code=None, *, retryable=False):
        self.status_code = status
        self.code = code
        self.retryable = retryable
        super().__init__(f'Azure HTTP {status}' + (f' ({code})' if code else ''))


def endpoint(client):
    if client.api_format != 'chat':
        raise ValueError('Azure transport currently supports Chat Completions only')
    base = client.base_url.rstrip('/')
    path = parse.urlparse(base).path
    expected = '/openai/deployments/' + parse.quote(client.model, safe='') + '/chat/completions'
    if path not in ('', expected):
        raise ValueError('Azure base URL must be the resource root or matching deployment endpoint')
    return (base if path else base+expected)+'?'+parse.urlencode({'api-version': configuration()['api_version']})


def token_estimate(payload):
    def size(value):
        if isinstance(value, dict):
            if value.get('type') == 'image_url':
                return 768 * 3  # Conservative image reservation, excluding base64 length.
            return sum(size(v) for v in value.values())
        if isinstance(value, list):
            return sum(size(v) for v in value)
        return len(str(value))
    return int(size(payload.get('messages', []))/3)+int(payload.get('max_completion_tokens',8192))+128


class RateLimiter:
    def __init__(self, database=None, config_path=None):
        self.path = Path(database or os.environ['AZURE_RATE_LIMIT_DB'])
        self.config_path = config_path or os.environ.get('AZURE_RATE_LIMIT_CONFIG')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS reservations (id TEXT PRIMARY KEY, started REAL, tokens INTEGER, active INTEGER, pid INTEGER)')
            db.execute('CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value REAL)')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute('PRAGMA busy_timeout=30000')
            with db:
                yield db
        finally:
            db.close()

    def limits(self):
        d = json.loads(Path(self.config_path).read_text()) if self.config_path else {}
        result = {'tokens_per_minute': d.get('tokens_per_minute',250000),
                  'requests_per_minute': d.get('requests_per_minute',120),
                  'max_concurrent': d.get('max_concurrent',24)}
        if any(type(v) is not int or v <= 0 for v in result.values()):
            raise ValueError('Azure operational limits must be positive integers')
        return result

    def acquire(self, estimate, timeout=1200):
        started = time.monotonic()
        while True:
            limits = self.limits()
            if estimate > limits['tokens_per_minute']:
                raise ValueError('single Azure request exceeds configured token reservation budget')
            now = time.time()
            with self.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                # Clean dead workers without leaking shared concurrency slots.
                for rid, pid in db.execute('SELECT id,pid FROM reservations WHERE active=1').fetchall():
                    try:
                        os.kill(pid,0)
                    except ProcessLookupError:
                        db.execute('UPDATE reservations SET active=0 WHERE id=?',(rid,))
                db.execute('DELETE FROM reservations WHERE active=0 AND started<?',(now-3600,))
                cooldown = db.execute("SELECT value FROM state WHERE key='cooldown'").fetchone()
                tokens, count = db.execute('SELECT COALESCE(SUM(tokens),0),COUNT(*) FROM reservations WHERE started>? OR active=1',(now-60,)).fetchone()
                active = db.execute('SELECT COUNT(*) FROM reservations WHERE active=1').fetchone()[0]
                if (not cooldown or now >= cooldown[0]) and tokens+estimate <= limits['tokens_per_minute'] and count < limits['requests_per_minute'] and active < limits['max_concurrent']:
                    rid = uuid.uuid4().hex
                    db.execute('INSERT INTO reservations VALUES (?,?,?,?,?)',(rid,now,estimate,1,os.getpid()))
                    return rid, time.monotonic()-started, limits
            if time.monotonic()-started > timeout:
                raise TimeoutError('timed out waiting for Azure experiment quota')
            time.sleep(.5+random.random()*.25)

    def release(self, rid, actual=None):
        with self.connect() as db:
            if actual is None:
                db.execute('UPDATE reservations SET active=0 WHERE id=?',(rid,))
            else:
                db.execute('UPDATE reservations SET active=0,tokens=? WHERE id=?',(max(0,int(actual)),rid))

    def pause(self, seconds):
        with self.connect() as db:
            db.execute("INSERT INTO state VALUES ('cooldown',?) ON CONFLICT(key) DO UPDATE SET value=MAX(value,excluded.value)",(time.time()+max(1,seconds),))


def _append_ledger(value):
    path = os.environ.get('AZURE_TRANSPORT_LEDGER_DIR')
    if path:
        directory = Path(path)
        directory.mkdir(parents=True,exist_ok=True)
        # One short atomic append per process/thread, without prompts or tokens.
        line = (json.dumps(value,ensure_ascii=False)+'\n').encode()
        fd = os.open(directory/(str(os.getpid())+'.jsonl'),os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
        try:
            os.write(fd,line)
        finally:
            os.close(fd)


def generate(client, payload):
    limiter = RateLimiter()
    encoded = json.dumps(payload,allow_nan=False).encode()
    signature = hashlib.sha256(encoded).hexdigest()
    for attempt in range(1,7):
        # Get/refresh credentials before taking an inference slot.
        token = access_token()
        rid, waited, limits = limiter.acquire(token_estimate(payload))
        began = time.monotonic()
        item = {'request_hash':signature,'time_unix':time.time(),'attempt':attempt,
                'configuration':configuration(),'model':client.model,'throttle_wait_seconds':waited,'limits':limits}
        actual = None
        retry_delay = 0
        try:
            req = request.Request(endpoint(client),data=encoded,headers={'Content-Type':'application/json','Authorization':'Bearer '+token},method='POST')
            with request.urlopen(req,timeout=client.timeout) as response:
                raw = json.load(response)
                headers = {k.lower():v for k,v in response.headers.items() if 'ratelimit' in k.lower() or k.lower() in ('retry-after','x-ms-request-id')}
            usage = raw.get('usage') or {}
            actual = usage.get('total_tokens')
            item.update(status='ok',http_status=200,usage=usage,response_model=raw.get('model'),headers=headers)
            return raw
        except error.HTTPError as exc:
            status = exc.code
            code = None
            try:
                body = json.loads(exc.read())
                value = body.get('error',{}).get('code')
                if isinstance(value,str) and len(value)<100 and all(c.isalnum() or c in '._-' for c in value):code=value
            except (ValueError,AttributeError,TypeError):
                pass
            retryable = status in (429,500,502,503,504)
            item.update(status='error',http_status=status,error_code=code)
            if status in (401,403):
                raise AzureError(status,code,retryable=False) from None
            if not retryable or attempt == 6:
                raise AzureError(status,code,retryable=False) from None
            try:
                retry_delay = float(exc.headers.get('retry-after') or exc.headers.get('retry-after-ms','0'))
                if not exc.headers.get('retry-after') and exc.headers.get('retry-after-ms'):retry_delay/=1000
            except ValueError:
                retry_delay = 0
            retry_delay = max(retry_delay,min(2**attempt,30))+random.random()
            limiter.pause(retry_delay)
        except (OSError,TimeoutError) as exc:
            item.update(status='error',error_type=type(exc).__name__)
            if attempt == 6:raise
            retry_delay = min(2**attempt,30)+random.random()
        finally:
            limiter.release(rid,actual)
            item['elapsed_seconds'] = time.monotonic()-began
            _append_ledger(item)
        time.sleep(retry_delay)
    raise RuntimeError('Azure retries exhausted')
