"""Download official visual sources, with resumable byte-range chunks."""
import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import time
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/openjev-vision/source'
CLEVR = 'https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip'
URLS = {
    'validation-images.csv': 'https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv',
    'train-images.csv': 'https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv',
    'validation-labels.csv': 'https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels.csv',
    'train-labels.csv': 'https://storage.googleapis.com/openimages/v5/train-annotations-human-imagelabels.csv',
    'classes.csv': 'https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv',
    'clevr-license.html': 'https://cs.stanford.edu/people/jcjohns/clevr/',
    'openimages-license.html': 'https://storage.googleapis.com/openimages/web/factsfigures.html',
}


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, url):
    dest = SOURCE/name
    if dest.exists():
        return
    for attempt in range(5):
        try:
            with requests.get(url, stream=True, timeout=(20, 180)) as r:
                r.raise_for_status()
                with dest.with_suffix(dest.suffix+'.part').open('wb') as f:
                    for chunk in r.iter_content(1024*1024):
                        f.write(chunk)
            os.replace(dest.with_suffix(dest.suffix+'.part'), dest)
            print(f'Downloaded {name}: {dest.stat().st_size:,} bytes', flush=True)
            return
        except requests.RequestException:
            if attempt == 4: raise
            time.sleep(2**attempt)


def clevr():
    dest = SOURCE/'CLEVR_selected.sparse.zip'
    if dest.exists(): return
    r = requests.get(CLEVR, headers={'Range':'bytes=0-0'}, timeout=30)
    r.raise_for_status()
    size = int(r.headers['Content-Range'].split('/')[-1])
    etag = r.headers['ETag']
    class Remote(io.RawIOBase):
        def __init__(self): self.pos = 0
        def seek(self, offset, whence=0):
            self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else size+offset
            return self.pos
        def tell(self): return self.pos
        def seekable(self): return True
        def read(self, n=-1):
            end = size-1 if n < 0 else min(size-1, self.pos+n-1)
            if end < self.pos: return b''
            response = requests.get(CLEVR,headers={'Range':f'bytes={self.pos}-{end}','If-Match':etag},timeout=120)
            response.raise_for_status()
            assert response.status_code == 206
            self.pos = end+1
            return response.content
    with zipfile.ZipFile(Remote()) as archive:
        entries = archive.infolist()
        train = [z for z in entries if '/images/train/' in z.filename and z.filename.endswith('.png')][:21000]
        val = [z for z in entries if '/images/val/' in z.filename and z.filename.endswith('.png')][:4000]
        extra = [z for z in entries if z.filename.endswith(('train_questions.json','val_questions.json','train_scenes.json','val_scenes.json'))]
        central = archive.start_dir
    assert len(train) == 21000 and len(val) == 4000 and len(extra) == 4
    selected = train+val+extra
    chunk_size = 64*1024*1024
    parts = SOURCE/'clevr-parts'; parts.mkdir(exist_ok=True)
    def part(start):
        end = min(size-1, start+chunk_size-1); path = parts/f'{start:012d}'
        if path.exists() and path.stat().st_size == end-start+1: return
        for attempt in range(5):
            try:
                with requests.get(CLEVR, headers={'Range':f'bytes={start}-{end}', 'If-Match':etag}, stream=True, timeout=(20,180)) as response:
                    if response.status_code != 206: raise ValueError('Range not honored')
                    with path.with_suffix('.part').open('wb') as f:
                        for chunk in response.iter_content(1024*1024): f.write(chunk)
                assert path.with_suffix('.part').stat().st_size == end-start+1
                path.with_suffix('.part').replace(path)
                return
            except requests.RequestException:
                if attempt == 4: raise
                time.sleep(2**attempt)
    starts = set(range((central//chunk_size)*chunk_size,size,chunk_size))
    for z in selected:
        starts.update(range((z.header_offset//chunk_size)*chunk_size,
                            min(size,z.header_offset+z.compress_size+1024),chunk_size))
    starts = sorted(starts)
    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        for i, _ in enumerate(ex.map(part, starts)):
            if (i+1) % 20 == 0: print(f'CLEVR chunks: {i+1}/{len(starts)}', flush=True)
    with dest.with_suffix('.part').open('wb') as f:
        f.truncate(size)
        for start in starts:
            f.seek(start)
            with (parts/f'{start:012d}').open('rb') as src:
                for chunk in iter(lambda: src.read(8*1024*1024), b''): f.write(chunk)
    dest.with_suffix('.part').replace(dest)
    (SOURCE/'clevr-http.json').write_text(json.dumps({'url':CLEVR,'size':size,'etag':etag,
        'storage':'sparse partial ZIP; only listed members and central directory are downloaded',
        'members':[{'name':z.filename,'crc32':z.CRC,'size':z.file_size} for z in selected]},indent=2)+'\n')
    print('CLEVR selected members complete', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--part', choices=['clevr','metadata'], required=True)
    args = p.parse_args(); SOURCE.mkdir(parents=True, exist_ok=True)
    if args.part == 'clevr': clevr()
    else:
        with concurrent.futures.ThreadPoolExecutor(4) as ex:
            list(ex.map(lambda pair: fetch(*pair), URLS.items()))
