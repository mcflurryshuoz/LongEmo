"""Check pinned media inventory and subtitle alignment without model calls."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import argparse
import json
from pathlib import Path
import subprocess

from evaluation.io_utils import write_json, validate_subtitles


def audit(root, output):
    manifest = json.loads((root/'manifest.json').read_text())
    questions = json.loads((root/'questions.json').read_text())
    video_ids = sorted({q['video_id'] for q in questions})
    files = {r['path']: r for r in manifest['files']}

    def inspect(vid):
        path = root/'episode/videos'/f'{vid}.mp4'
        subtitle = root/'prepared_subtitles'/f'{vid}.json'
        relative = str(path.relative_to(root))
        if not path.exists() or relative not in files or not subtitle.exists():
            return {'video_id': vid, 'status': 'missing'}
        if path.stat().st_size != files[relative]['bytes']:
            return {'video_id': vid, 'status': 'size_mismatch'}
        raw = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
            'format=duration:stream=codec_type,codec_name,start_time,duration,width,height', '-of', 'json', str(path)], timeout=30)
        data = json.loads(raw)
        video = next(s for s in data['streams'] if s['codec_type']=='video')
        audio = [s for s in data['streams'] if s['codec_type']=='audio']
        duration = float(video.get('duration', data['format']['duration']))
        rows = json.loads(subtitle.read_text())
        validate_subtitles(rows)
        outside = sum(r['t'][0] >= duration+2 or r['t'][1] > duration+2 for r in rows)
        return {'video_id': vid, 'status': 'ok' if audio else 'missing_audio',
                'video_duration_seconds': duration, 'video_start_seconds': float(video.get('start_time',0)),
                'audio_start_seconds': float(audio[0].get('start_time',0)) if audio else None,
                'video_codec': video.get('codec_name'), 'width': video.get('width'), 'height': video.get('height'),
                'subtitle_rows': len(rows), 'subtitle_end_seconds': max((r['t'][1] for r in rows),default=0),
                'subtitles_over_two_seconds_outside_video': outside,
                'subtitle_start_order_violations': sum(rows[i]['t'][0] < rows[i-1]['t'][0] for i in range(1,len(rows))),
                'bytes': files[relative]['bytes'], 'download_verified_sha256': files[relative]['sha256']}

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(inspect, video_ids))
    report = {'revision': manifest['revision'], 'question_count': len(questions), 'video_count': len(rows),
              'task_counts': dict(Counter(q['type'] for q in questions)),
              'reasoning_subtypes': dict(Counter('result' if max(map(int,q['rubric']['scores']))==1 else 'explanation'
                  for q in questions if q['type']=='emotional reasoning')),
              'statuses': dict(Counter(r['status'] for r in rows)),
              'videos_with_subtitle_overrun': [r['video_id'] for r in rows if r.get('subtitles_over_two_seconds_outside_video',0)],
              'videos_with_subtitle_order_issues': [r['video_id'] for r in rows if r.get('subtitle_start_order_violations',0)],
              'invalid_public_questions': [q['question_id'] for q in questions if not isinstance(q.get('question'),str) or not q['question'].strip()],
              'integrity_note': 'Download verifies byte count and available HF LFS SHA256; this audit checks byte count, streams, and converted subtitle schema/timing. It does not independently rehash all video bytes.',
              'videos': rows}
    write_json(output,report)
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.data_root,a.output)
    print(json.dumps({k:v for k,v in r.items() if k!='videos'}))
