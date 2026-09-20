"""Package verified game decisions, frozen inputs, licenses, and rebuild tooling."""
import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(directory, source):
    report = json.loads((directory / 'report.json').read_text())
    tokenizer = json.loads((directory / 'tokenizer_audit.json').read_text())
    audit = json.loads((directory / 'game_audit.json').read_text())
    repro = json.loads((directory / 'reproducibility.json').read_text())
    assert repro['all_equal'] and audit['records'] == report['validation']['records']
    assert sha(ROOT / 'scripts/build_games_dataset.py') == report['build_script_sha256']
    assert sha(source / 'sources.lock.json') == report['sources_lock_sha256']
    for split in ('train', 'validation', 'calibration'):
        path = directory / (split + '.jsonl')
        assert sha(path) == report['splits'][split]['sha256'] == tokenizer['splits'][split]['jsonl_sha256']
        original = [json.loads(line) for line in path.open()]
        restored = pq.read_table(directory / (split + '.parquet')).to_pylist()
        assert len(original) == len(restored)
        for a, b in zip(original, restored):
            for key in ('questions', 'targets', 'provenance', 'acceptable_targets', 'target_distributions'):
                if b.get(key) is not None:
                    b[key] = json.loads(b[key])
            assert a == {k: v for k, v in b.items() if v is not None}
    shutil.copytree(source, directory / 'source', dirs_exist_ok=True)
    shutil.copytree(ROOT / 'dataset/games', directory / 'dataset/games', dirs_exist_ok=True)
    for folder, names in {
        'scripts': ['build_games_dataset.py', 'download_game_sources.py', 'audit_games_dataset.py',
                    'audit_public_dataset.py', 'package_games_dataset.py', 'verify_games_rebuild.py'],
        'tests': ['test_games_dataset.py'],
        'openjev': [p.name for p in (ROOT / 'openjev').glob('*.py')],
    }.items():
        (directory / folder).mkdir(exist_ok=True)
        for name in names:
            shutil.copy2(ROOT / folder / name, directory / folder / name)
    for name in ('LICENSE', 'pyproject.toml'):
        shutil.copy2(ROOT / name, directory / name)
    shutil.copy2(ROOT / 'dataset/games/LICENSE-DATA.md', directory / 'LICENSE-DATA.md')
    shutil.copy2(ROOT / 'dataset/requirements-build.txt', directory / 'dataset/requirements-build.txt')
    card = '''---
language: [en]
license: other
license_name: mixed-permissive-cc0-apache
license_link: LICENSE-DATA.md
task_categories: [text-classification]
size_categories: [10K<n<100K]
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.parquet
      - split: validation
        path: validation.parquet
      - split: calibration
        path: calibration.parquet
---
# OpenJev games v1

45,300 decisions: 25,000 chess, 15,000 tactical Gomoku, 5,000 exact Connect Four
endgames, and 300 distinct Kuhn/Leduc poker information states. No Gemini or paid
generation. 40,785 train / 2,246 validation / 2,269 calibration.

Read [the methodology](dataset/games/METHODOLOGY.md) and
[license notice](LICENSE-DATA.md) before use. JSONL is the canonical format.
Parquet dictionary columns contain JSON strings and must be parsed after loading.
Only `state` and `questions` are model inputs. Proofs and labels are metadata.
Poker probabilities are **mixed strategies**, not confidence; use soft targets
and sample actions when playing. Exclude poker from confidence calibration.

This is a dataset release candidate, not a trained model or evidence of improved
playing strength. Gomoku covers short tactics; Connect Four covers endgames.
Chess engine labels are bounded-search approximations. Source samples are frozen
prefixes, not unbiased samples of all Lichess positions. Internal held-out splits
are not independent public benchmarks.

For an offline rebuild from this archive, install
`dataset/games/requirements-build.txt`, then run:

```bash
python scripts/build_games_dataset.py --source source --output rebuilt
python scripts/audit_games_dataset.py --dataset rebuilt
pytest -q tests/test_games_dataset.py
```

The separate tokenizer audit requires the named tokenizer to be locally cached
and the additional packages in `dataset/requirements-build.txt`.
All frozen chess inputs, source checksums, build code, tests, and audit reports
are included. `SHA256SUMS` covers every archive file except itself.
'''
    (directory / 'README.md').write_text(card)
    files = sorted(p for p in directory.rglob('*') if p.is_file() and p.name != 'SHA256SUMS' and '__pycache__' not in p.parts)
    (directory / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(directory)}\n' for p in files))
    archive = directory.parent / 'openjev-games-v1.tar.gz'
    with archive.open('wb') as out, gzip.GzipFile(fileobj=out, mode='wb', filename='', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as tar:
            for path in sorted(files + [directory / 'SHA256SUMS']):
                info = tar.gettarinfo(str(path), arcname='openjev-games-v1/' + str(path.relative_to(directory)))
                info.mtime = 0; info.uid = info.gid = 0; info.uname = info.gname = ''; info.mode = 0o644
                with path.open('rb') as f:
                    tar.addfile(info, f)
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive)}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=ROOT / 'data/openjev-games/release')
    p.add_argument('--source', type=Path, default=ROOT / 'data/openjev-games/source')
    args = p.parse_args()
    package(args.dataset, args.source)
