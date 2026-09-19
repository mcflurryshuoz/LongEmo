"""Independent one-window BlackAI frontend diagnostic; never adds benchmark scores."""
import argparse
import json
from pathlib import Path
import time

from evaluation.clients import Client
from evaluation.inference.adapters import gemini
from evaluation.io_utils import write_json
from experiments.zyf.native_worker import observed_parser
from methods.longemo.audio import bridge_audio
from methods.longemo.common import LoggedClient, code_hash, file_hash
from methods.longemo.media import probe, subtitles, window_input
from methods.longemo.memory import empty_memory, apply_window
from methods.longemo.prompts import PERCEPTION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--credential-file', type=Path, required=True)
    p.add_argument('--video-id', default='G2_V000016')
    args = p.parse_args()
    start = time.time()
    out = args.runtime/'diagnostics/blackai_recheck_20260919'/str(time.time_ns())
    out.mkdir(parents=True)
    url = 'https://www.blackaicoding.com/v1beta'
    model = 'gemini-3.8-flash'
    key = json.loads(args.credential_file.read_text())['GEMINI_API_KEY']
    visual = Client(model, url, 'gemini', key, 240, 32768, 1,
                    {'generationConfig': {'thinkingConfig': {'thinkingLevel': 'medium'}}})
    audio = Client(model, url, 'gemini', key, 180, 4096, None,
                   {'generationConfig': {'thinkingConfig': {'thinkingLevel': 'low'}}})
    original = gemini.parse_response
    gemini.parse_response = observed_parser(original, out/'native_responses.jsonl')
    result = {'host': 'g450', 'time_unix': start, 'video_id': args.video_id,
              'core_interval': [0, 20], 'media_interval': [0, 22],
              'model': model, 'base_url': url, 'method_hash': code_hash(),
              'selection': 'First window of existing development video V16; no selection by rejected content.',
              'scope': 'Independent frontend diagnostic, not a benchmark score or complete memory.',
              'media': {'fps': 1, 'max_frames': 24, 'max_pixels': 200704},
              'attempts_per_stage': 1}
    stage = 'media'
    try:
        root = args.runtime/'data'
        video = root/'episode/videos'/(args.video_id+'.mp4')
        sha = file_hash(video)
        manifest = json.loads((root/'manifest.json').read_text())
        entry = next(f for f in manifest['files'] if f['path'] == 'episode/videos/'+video.name)
        assert sha == entry['sha256'] and video.stat().st_size == entry['bytes']
        info = probe(video)
        assert info['duration'] >= 22
        result.update(video_sha256=sha, data_revision=manifest['revision'])
        memory = empty_memory(args.video_id, info['duration'], sha)
        content, metadata = window_input(video, 0, 22, fps=1, max_frames=24, max_pixels=200704,
            with_audio=True, subtitle_rows=subtitles(root/'prepared_subtitles'/(args.video_id+'.json')))
        stage = 'audio'
        content, trace = bridge_audio(content, [0, 22], audio, out/'audio/W00001.json', tries=1)
        metadata.update(audio_representation='derived_timestamped_cues', audio_observer=trace)
        context = {'video_id': args.video_id, 'window_id': 'W00001', 'core_interval': [0, 20],
                   'media_interval': [0, 22], 'corrections_enabled': False,
                   'cast': memory['entities'], 'preceding_events': []}
        messages = [{'role': 'system', 'content': PERCEPTION}, {'role': 'user', 'content':
            [{'type': 'text', 'text': json.dumps(context, ensure_ascii=False)}] + content}]
        def validate(payload):
            return apply_window(memory, payload, window_id='W00001', core=[0, 20],
                                media=[0, 22], metadata=metadata, allow_revisions=False)
        stage = 'perception'
        payload = LoggedClient(visual, out/'calls.jsonl', tries=1).call(
            messages, purpose='perception:'+args.video_id+':W00001', validate=validate)
        updated = validate(payload)
        write_json(out/'window.json', {'input': context, 'sampling': metadata, 'perception': payload})
        result.update(status='ok', validated_windows=1, entities=len(updated['entities']),
                      events=len(updated['events']),
                      audio_observations=len(json.loads((out/'audio/W00001.json').read_text())['result']['observations']))
    except Exception as exc:
        result.update(status='error', failed_stage=stage, error_type=type(exc).__name__,
                      error=str(exc).replace(key, '<redacted>')[:400])
    finally:
        gemini.parse_response = original
    result['elapsed_seconds'] = round(time.time()-start, 3)
    result['diagnostic_dir'] = str(out)
    write_json(out/'result.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
