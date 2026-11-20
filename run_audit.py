"""Provenance and effective-mode checks; no market/network dependencies."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from execution_policy import get_policy


def resolve_entry_mode(environment):
    mode = environment.get('ENTRY_MODE', 'balanced').strip().lower()
    get_policy(mode)
    expected = environment.get('RUN_EXPECTED_ENTRY_MODE', '').strip().lower()
    if expected and mode != expected:
        raise ValueError(f'ENTRY_MODE={mode}, expected {expected}; check the quant bot step env')
    return mode


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def frame_sha256(frame):
    text = frame.to_csv(index=True, date_format='%Y-%m-%d', float_format='%.17g')
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def package_versions():
    result = {'python': platform.python_version()}
    for package in ('pandas', 'numpy', 'requests', 'xgboost', 'scikit-learn'):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def build_manifest(root, mode, model_files, frames, effective_config, loaded, missing):
    root = Path(root)
    paths = ['quant_bot.py', 'execution_policy.py', 'run_audit.py', 'vn30_universe.py',
             'data/markets_database10years.csv'] + list(model_files)
    return {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'entry_mode': mode,
        'entry_policy': asdict(get_policy(mode)),
        'effective_config': effective_config,
        'sha256': {name: sha256_file(root/name) for name in sorted(set(paths)) if (root/name).is_file()},
        'input_frames_sha256': {name: frame_sha256(frame) for name, frame in frames.items()},
        'versions': package_versions(),
        'loaded_tickers': sorted(loaded),
        'missing_tickers': sorted(missing),
        'modeled_order_fills': True,
        'ai_target': '30-session return above VNINDEX; AI score is not completed-trade win probability',
        'historical_claim': 'Annual walk-forward; data availability and strategy selection still require validation',
    }


def write_manifest(path, manifest):
    Path(path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
