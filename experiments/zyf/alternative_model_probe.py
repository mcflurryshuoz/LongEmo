"""Single-attempt model comparison on fixed failed windows; never alters benchmark runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
from urllib.parse import urlparse

from evaluation.inference.adapters import anthropic, gemini
from experiments.zyf.aicodemirror_probe import BASE, mirror_headers, probe_video
from experiments.zyf.blackai_continuation import read, atomic

CLAUDE_BASE = "https://api.aicodemirror.ai/api/claudecode/v1"
ORIGINAL_ANTHROPIC_HEADERS = anthropic.headers
PROFILES = {
    "blackai_gemini25_pro": {"provider": "BlackAI", "model": "gemini-2.5-pro", "base_url": "https://www.blackaicoding.com/v1beta", "api_format": "gemini",
        "audio_options": {"generationConfig": {"thinkingConfig": {"thinkingBudget": 1024}}},
        "visual_options": {"generationConfig": {"thinkingConfig": {"thinkingBudget": 1024}}}},
    "mirror_gemini25_pro": {"model": "gemini-2.5-pro", "base_url": BASE, "api_format": "gemini",
        "audio_options": {"generationConfig": {"thinkingConfig": {"thinkingBudget": 1024}}},
        "visual_options": {"generationConfig": {"thinkingConfig": {"thinkingBudget": 1024}}}},
    "mirror_gemini31_pro": {"model": "gemini-3.1-pro", "base_url": BASE, "api_format": "gemini",
        "audio_options": {"generationConfig": {"thinkingConfig": {"thinkingLevel": "low"}}},
        "visual_options": {"generationConfig": {"thinkingConfig": {"thinkingLevel": "low"}}}},
    "mirror_claude_opus5": {"model": "claude-opus-5", "base_url": CLAUDE_BASE, "api_format": "anthropic"},
    "mirror_claude_sonnet5": {"model": "claude-sonnet-5", "base_url": CLAUDE_BASE, "api_format": "anthropic"},
}


def claude_headers(client):
    if urlparse(client.base_url).hostname != "api.aicodemirror.ai":
        return ORIGINAL_ANTHROPIC_HEADERS(client)
    return {"Content-Type": "application/json", "anthropic-version": "2023-06-01",
            "Authorization": "Bearer " + client.api_key}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--credential-file', type=Path, required=True)
    parser.add_argument('--profile', choices=sorted(PROFILES), action='append', required=True)
    parser.add_argument('--video-id', action='append', required=True)
    args = parser.parse_args()
    assert set(args.video_id) <= {f'G2_V{x:06d}' for x in [48,71,104,105,106,108,110,112,113,115,116,117,129,131,137]}
    assert len({PROFILES[name].get('provider', 'AICodeMirror') for name in args.profile}) == 1, 'use separate credentials for each provider'
    key = read(args.credential_file)['GEMINI_API_KEY']
    root = args.runtime/'diagnostics/alternative_models_20260921'
    root.mkdir(exist_ok=True)
    gemini.headers = mirror_headers
    anthropic.headers = claude_headers
    jobs = []
    for name in args.profile:
        folder = root/name;folder.mkdir(exist_ok=True)
        profile = {'provider':'AICodeMirror', **PROFILES[name]}
        if (folder/'profile.json').exists():
            assert read(folder/'profile.json') == profile, 'profile changed after first probe'
        else:
            atomic(folder/'profile.json', profile)
        for vid in args.video_id:
            jobs.append((name, folder, profile, vid))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(probe_video,args.runtime,folder,key,vid,profile):name
                   for name,folder,profile,vid in jobs}
        for future in as_completed(futures):
            row = future.result()
            print(json.dumps({k:v for k,v in row.items() if k not in ['usage','client_configuration']},ensure_ascii=False),flush=True)
    rows = [{**read(p), 'profile':p.parent.parent.name} for p in root.glob('*/*/result.json')]
    atomic(root/'summary.json', {'updated_unix':time.time(), 'scope':'One original failed window per video/model; no graph completion or benchmark scores', 'results':rows})


if __name__ == '__main__':
    main()
