"""Verify and package an audited Hold'em build and an independent rebuild."""
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


def package(directory, rebuild):
    if directory.resolve() == rebuild.resolve():
        raise ValueError('An independent build directory is required')
    report = json.loads((directory/'report.json').read_text())
    other = json.loads((rebuild/'report.json').read_text())
    audit = json.loads((directory/'audit.json').read_text())
    tokenizer = json.loads((directory/'tokenizer_audit.json').read_text())
    for key in ('compiler_sha256', 'seed', 'versions', 'records'):
        assert report[key] == other[key], key
    assert report['compiler_sha256'] == sha(ROOT/'scripts/build_holdem_dataset.py')
    assert report['records'] == audit['records']
    evidence = {}
    for split in ('train', 'validation', 'calibration'):
        expected = report['splits'][split]['sha256']
        assert expected == audit['split_sha256'][split] == tokenizer['splits'][split]['jsonl_sha256']
        for ext in ('jsonl', 'parquet'):
            name = split+'.'+ext
            a, b = sha(directory/name), sha(rebuild/name)
            assert a == b, name
            if ext == 'jsonl':
                assert a == expected
            evidence[name] = {'sha256': a, 'independent_build_sha256': b, 'equal': True}
        original = [json.loads(line) for line in (directory/(split+'.jsonl')).open()]
        restored = pq.read_table(directory/(split+'.parquet')).to_pylist()
        assert len(original) == len(restored)
        for a, b in zip(original, restored):
            for key in ('questions', 'targets', 'provenance', 'target_distributions'):
                if b.get(key) is not None:
                    b[key] = json.loads(b[key])
            assert a == {k: v for k, v in b.items() if v is not None}
    repro = {'all_equal': True, 'files': evidence, 'scope': 'two independent builds in pinned environment'}
    (directory/'reproducibility.json').write_text(json.dumps(repro, indent=2)+'\n')
    shutil.copytree(ROOT/'dataset/holdem', directory/'dataset/holdem', dirs_exist_ok=True)
    for folder, names in {
        'scripts': ['build_holdem_dataset.py', 'audit_holdem_dataset.py', 'package_holdem_dataset.py', 'audit_public_dataset.py'],
        'tests': ['test_holdem_dataset.py'],
        'openjev': [p.name for p in (ROOT/'openjev').glob('*.py')],
    }.items():
        (directory/folder).mkdir(exist_ok=True)
        for name in names:
            shutil.copy2(ROOT/folder/name, directory/folder/name)
    for name in ('LICENSE', 'pyproject.toml'):
        shutil.copy2(ROOT/name, directory/name)
    shutil.copy2(ROOT/'dataset/requirements-build.txt', directory/'dataset/requirements-build.txt')
    shutil.copy2(ROOT/'dataset/holdem/LICENSE-DATA.md', directory/'LICENSE-DATA.md')
    (directory/'README.md').write_text('''---
language: [en]
license: apache-2.0
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
# OpenJev Texas Hold'em v1

15,000 procedural decisions: 4,000 hand categories, 3,000 category-improvement
draw counts, 3,000 exact showdown equity questions, and 5,000 strategy decisions
from 625 certified **restricted single-bet river games**. These are not full-game
no-limit Hold'em strategies. No Gemini, paid API, or proprietary data is used.

13,623 train / 635 validation / 742 calibration. JSONL is canonical. Nested
Parquet columns are JSON strings. Only `state` and `questions` are model inputs;
provenance, solver policies, and labels must not be put in prompts.

River `target_distributions` describe mixed actions, not confidence. Use soft
targets and sample policies at play time. Exclude these strategy rows from
confidence-temperature calibration, even when stored in the calibration file.
Fundamental equity labels describe showdown equity under explicit ranges; they
are not betting advice or action labels. No model quality claim is made.

Read [methodology](dataset/holdem/METHODOLOGY.md) and [license notes](LICENSE-DATA.md).
Independent audits, reproducibility evidence, and file checksums are included.
To rebuild from the extracted archive:

```bash
pip install -r dataset/holdem/requirements-build.txt
python scripts/build_holdem_dataset.py --output rebuilt
python scripts/audit_holdem_dataset.py --dataset rebuilt
pytest -q tests/test_holdem_dataset.py
```

The optional tokenizer audit additionally requires the cached tokenizer named
in its report and packages in `dataset/requirements-build.txt`. Model assets
and dependency packages are not redistributed in this archive.
''')
    files = sorted(p for p in directory.rglob('*') if p.is_file() and p.name != 'SHA256SUMS' and '__pycache__' not in p.parts)
    (directory/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(directory)}\n' for p in files))
    archive = directory.parent/'openjev-holdem-v1.tar.gz'
    with archive.open('wb') as dest, gzip.GzipFile(fileobj=dest, mode='wb', filename='', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as tar:
            for path in sorted(files+[directory/'SHA256SUMS']):
                info = tar.gettarinfo(str(path), arcname='openjev-holdem-v1/'+str(path.relative_to(directory)))
                info.uid = info.gid = info.mtime = 0; info.uname = info.gname = ''; info.mode = 0o644
                with path.open('rb') as f:
                    tar.addfile(info, f)
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive)}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=ROOT/'data/openjev-holdem/release')
    p.add_argument('--rebuild', type=Path, required=True)
    args = p.parse_args()
    package(args.dataset, args.rebuild)
