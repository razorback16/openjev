"""Freeze selected annotations; restore offline build inputs from a full release."""
import argparse
import gzip
import json
import shutil
import zipfile
from pathlib import Path

import build_vision_pool as build


def freeze():
    output=build.RELEASE/'sources';output.mkdir(exist_ok=True)
    selected=build.select_games()
    with gzip.open(output/'games.selected.jsonl.gz','wt') as f:
        for row in selected:f.write(json.dumps(row,separators=(',',':'))+'\n')
    shutil.copy2(output/'games.selected.jsonl.gz',build.SOURCE/'games.selected.jsonl.gz')
    for name in ('clevr-http.json','clevr-license.html','openimages-license.html'):
        shutil.copy2(build.SOURCE/name,output/name)
    shutil.copy2(build.SOURCE/'photos/selection.jsonl',output/'photos.selection.jsonl')
    quality=build.SOURCE/'photos/selection-quality-report.json'
    if quality.exists():shutil.copy2(quality,output/quality.name)
    manifest=json.loads((build.SOURCE/'clevr-http.json').read_text())
    filenames={Path(x['name']).name for x in manifest['members'] if x['name'].endswith('.png')}
    annotations={}
    with zipfile.ZipFile(build.SOURCE/'CLEVR_selected.sparse.zip') as archive:
        for split in ('train','val'):
            for kind in ('questions','scenes'):
                name=f'CLEVR_v1.0/{kind}/CLEVR_{split}_{kind}.json'
                obj=json.loads(archive.read(name))
                obj[kind]=[x for x in obj[kind] if x['image_filename'] in filenames]
                annotations[name]=obj
    with gzip.open(output/'clevr.annotations.json.gz','wt') as f:json.dump(annotations,f,separators=(',',':'))
    print('Selected source snapshots frozen',flush=True)


def restore(release,output):
    output.mkdir(parents=True,exist_ok=True)
    for name in ('games.selected.jsonl.gz','clevr-http.json','clevr-license.html','openimages-license.html'):
        shutil.copy2(release/'sources'/name,output/name)
    (output/'photos/images').mkdir(parents=True,exist_ok=True)
    shutil.copy2(release/'sources/photos.selection.jsonl',output/'photos/selection.jsonl')
    rows=[json.loads(line) for split in build.SPLITS for line in (release/f'{split}.jsonl').open()]
    with zipfile.ZipFile(output/'CLEVR_selected.sparse.zip','w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
        for row in rows:
            p=row['provenance'];image=release/row['images'][0]['path']
            if p['family']=='clevr':
                archive.write(image,f'CLEVR_v1.0/images/{p["original_split"]}/{p["source_image"]}')
            elif p['family']=='photos':
                shutil.copyfile(image,output/'photos/images'/f'{p["source_image_id"]}.jpg')
        with gzip.open(release/'sources/clevr.annotations.json.gz','rt') as f:annotations=json.load(f)
        for name,obj in annotations.items():archive.writestr(name,json.dumps(obj,separators=(',',':')))
    print('Offline source inputs restored from release',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--restore',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.restore:
        if args.output is None:p.error('--output is required for restoration')
        restore(args.restore,args.output)
    else:freeze()
