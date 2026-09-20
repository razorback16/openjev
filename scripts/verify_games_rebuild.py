"""Compare two independent builds and record release reproducibility evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(reference, other):
    if reference.resolve() == other.resolve():
        raise ValueError('Supply two different build directories')
    files = {}
    for split in ('train', 'validation', 'calibration'):
        for extension in ('jsonl', 'parquet'):
            name = split + '.' + extension
            a = hashlib.sha256((reference / name).read_bytes()).hexdigest()
            b = hashlib.sha256((other / name).read_bytes()).hexdigest()
            if a != b:
                raise ValueError(f'Builds differ: {name}')
            files[name] = {'sha256': a, 'independent_build_sha256': b, 'equal': True}
    for key in ('seed', 'versions', 'sources_lock_sha256', 'build_script_sha256'):
        if json.loads((reference / 'report.json').read_text())[key] != json.loads((other / 'report.json').read_text())[key]:
            raise ValueError(f'Build provenance differs: {key}')
    report = {'all_equal': True, 'files': files, 'seed': json.loads((reference / 'report.json').read_text())['seed'],
              'scope': 'two independent offline builds with the same frozen inputs and pinned environment'}
    (reference / 'reproducibility.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Both builds match: all six dataset files')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, default=Path('data/openjev-games/release'))
    p.add_argument('--other', type=Path, required=True)
    args = p.parse_args()
    verify(args.reference, args.other)
