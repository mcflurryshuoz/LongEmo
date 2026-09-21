"""BlackAI Pro profile and a reusable one-window gate, without core-method edits."""
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time

from evaluation.clients import Client
from evaluation.io_utils import write_json
from experiments.zyf.blackai_continuation import BLACKAI, read
from experiments.zyf.native_worker import main
from methods.longemo import runner
from methods.longemo.audio import bridge_audio
from methods.longemo.common import LoggedClient, code_hash, file_hash, git_revision, manifest, fingerprint
from methods.longemo.memory import apply_window
from methods.longemo.prompts import PERCEPTION

MODEL = 'gemini-3.1-pro-preview'
ORIGINAL_MEDIA = runner.window_input


def visual_client(args):
    assert args.model == MODEL and args.base_url == BLACKAI
    assert args.max_tokens == 8192 and args.thinking == 'default'
    key = read(Path(args.credential_file))['GEMINI_API_KEY']
    return Client(MODEL, BLACKAI, 'gemini', key, 240, 8192, None, {})


def audio_client(args):
    assert args.with_audio and args.audio_model == MODEL and args.audio_base_url == BLACKAI
    key = read(Path(args.credential_file))['GEMINI_API_KEY']
    return Client(MODEL, BLACKAI, 'gemini', key, 180, 4096, None, {})


def bounded_media(*args, **kwargs):
    # Two simultaneous decoders; this changes scheduling, never sampled media.
    root = Path(os.environ['LONGEMO_MEDIA_LOCK_DIR']); root.mkdir(parents=True, exist_ok=True)
    while True:
        for slot in range(2):
            lock = (root / ('slot%d.lock' % slot)).open('a')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close(); continue
            try:
                return ORIGINAL_MEDIA(*args, **kwargs)
            finally:
                lock.close()
        time.sleep(0.2)


def probe_one(video_id, args, client):
    """Commit exactly one validated window; the ordinary builder reuses it."""
    folder = Path(args.output_dir) / video_id
    assert not (folder / 'probe_attempt.json').exists(), 'one-window gate already attempted'
    video = Path(args.videos_dir) / (video_id + '.mp4')
    video_sha = file_hash(video)
    info = runner.probe(video)
    subtitle_path = Path(args.subtitles_dir) / (video_id + '.json')
    rows = runner.subtitles(subtitle_path)
    audio = audio_client(args)
    config = {'audio_observer': audio.configuration(), 'method': 'longemo-joint-perception-v1',
              'video_sha256': video_sha, 'subtitles_sha256': file_hash(subtitle_path),
              'model': client.configuration(), 'window_seconds': args.window_seconds, 'padding': args.padding,
              'media': runner._media_options(args), 'allow_revisions': args.allow_revisions,
              'code_hash': code_hash(), 'git_revision': git_revision()}
    build_hash = manifest(folder / 'manifest.json', config)
    memory = read(folder / 'memory.json'); assert not memory.get('complete')
    assert memory['video_sha256'] == video_sha
    memory['build_fingerprint'] = build_hash
    count = math.ceil(info['duration'] / args.window_seconds)
    i = next(i for i in range(count) if f'W{i+1:05d}' not in memory['completed_windows'])
    wid = f'W{i+1:05d}'
    core = [i * args.window_seconds, min((i+1) * args.window_seconds, info['duration'])]
    interval = [max(0, core[0]-args.padding), min(info['duration'], core[1]+args.padding)]
    evidence = {'video_id': video_id, 'window_id': wid, 'attempts_per_stage': 1,
                'started_unix': time.time(), 'stage': 'media', 'parent_memory_sha256': file_hash(folder/'memory.json')}
    write_json(folder/'probe_attempt.json', evidence)
    media, metadata = runner.window_input(video, *interval, subtitle_rows=rows, **runner._media_options(args))
    evidence['stage'] = 'audio'; write_json(folder/'probe_attempt.json', evidence)
    media, trace = bridge_audio(media, interval, audio, folder/'audio'/(wid+'.json'), tries=1)
    metadata.update(audio_representation='derived_timestamped_cues', audio_observer=trace)
    context = {'video_id':video_id, 'window_id':wid, 'core_interval':core, 'media_interval':interval,
               'corrections_enabled':args.allow_revisions, 'cast':memory['entities'], 'preceding_events':memory['events'][-3:]}
    messages = [{'role':'system','content':PERCEPTION}, {'role':'user','content':[
        {'type':'text','text':json.dumps(context,ensure_ascii=False)}]+media}]
    def validate(payload):
        return apply_window(memory,payload,window_id=wid,core=core,media=interval,
                            metadata=metadata,allow_revisions=args.allow_revisions)
    evidence.update(stage='visual', request_hash=fingerprint(messages)); write_json(folder/'probe_attempt.json',evidence)
    payload = LoggedClient(client,folder/'calls.jsonl',tries=1).call(messages,purpose=f'perception:{video_id}:{wid}',validate=validate)
    updated = validate(payload)
    updated['complete'] = len(updated['completed_windows']) == count
    write_json(folder/'windows'/(wid+'.json'),{'input':context,'sampling':metadata,'perception':payload})
    write_json(folder/'memory.json',updated)
    result = {**evidence,'status':'ok','finished_unix':time.time(),'complete':updated['complete'],
              'window_sha256':file_hash(folder/'windows'/(wid+'.json')),'memory_sha256':file_hash(folder/'memory.json')}
    write_json(folder/'probe_result.json',result)
    print(json.dumps({k:result[k] for k in ['video_id','window_id','status','complete']}),flush=True)
    return result


if __name__ == '__main__':
    probe_only = '--probe-only' in sys.argv
    if probe_only: sys.argv.remove('--probe-only'); runner._build_video = probe_one
    runner.client_for = visual_client
    runner.audio_client_for = audio_client
    runner.window_input = bounded_media
    raise SystemExit(main())
