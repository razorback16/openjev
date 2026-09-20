"""Freeze small, redistributable prefixes of the official CC0 Lichess exports."""
import argparse
import hashlib
import io
import json
import urllib.request
from pathlib import Path

import zstandard


def download(output, puzzles=60000, evaluations=120000):
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, count, header in [('puzzle.csv', puzzles, True), ('eval.jsonl', evaluations, False)]:
        url = 'https://database.lichess.org/lichess_db_' + name + '.zst'
        path = output / name
        if path.exists():
            raise FileExistsError(f'{path}: keep the frozen sources or use a new directory')
        with urllib.request.urlopen(url, timeout=120) as response:
            headers = dict(response.headers)
            reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(response))
            with path.open('w') as dest:
                for i in range(count + int(header)):
                    line = reader.readline()
                    if not line:
                        raise ValueError('Unexpected end of source')
                    dest.write(line)
        manifest.append({'file': name, 'url': url, 'license': 'CC0-1.0',
                         'extraction': 'first complete records, original order', 'records': count,
                         'http_headers': headers, 'bytes': path.stat().st_size,
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        print(f'Frozen {count} records: {path}', flush=True)
    (output / 'sources.lock.json').write_text(json.dumps(manifest, indent=2) + '\n')
    urllib.request.urlretrieve('https://database.lichess.org/', output / 'lichess-license-source.html')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('data/openjev-games/source'))
    args = p.parse_args()
    download(args.output)
