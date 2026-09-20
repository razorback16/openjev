"""Verify a second build and write split-specific WebDataset image/JSON shards."""
import argparse
import concurrent.futures
import copy
import gzip
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

import build_vision_pool as build

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def tar_bytes(tar,name,data):
    info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0;info.mode=0o644
    tar.addfile(info,io.BytesIO(data))


def write_shard(task):
    directory,split,index,rows=task
    path=directory/'shards'/f'{split}-{index:05d}.tar'
    with tarfile.open(path,'w',format=tarfile.PAX_FORMAT) as tar:
        for row in rows:
            record=copy.deepcopy(row);im=record['images'][0]
            data=(directory/im['path']).read_bytes()
            assert hashlib.sha256(data).hexdigest()==im['sha256']
            name=row['id']+Path(im['path']).suffix;im['path']=name
            tar_bytes(tar,name,data)
            tar_bytes(tar,row['id']+'.json',json.dumps(record,separators=(',',':')).encode())
    return {'file':str(path.relative_to(directory)),'split':split,'records':len(rows),'bytes':path.stat().st_size,'sha256':sha(path)}


def package(directory,rebuild):
    if directory.resolve()==rebuild.resolve():raise ValueError('Supply a separate rebuild directory')
    report=json.loads((directory/'report.json').read_text())
    assert report['images']==100000 and report['splits']['evaluation']['images']==10000
    assert report['builder_sha256']==sha(ROOT/'scripts/build_vision_pool.py')
    evidence={}
    for family in ('games','structured','clevr','photos'):
        name=f'parts/{family}.jsonl';a,b=sha(directory/name),sha(rebuild/name)
        assert a==b,f'Rebuild differs: {family}'
        evidence[name]={'sha256':a,'independent_build_sha256':b}
    rows={s:[json.loads(line) for line in (directory/f'{s}.jsonl').open()] for s in build.SPLITS}
    all_rows=[r for values in rows.values() for r in values]
    def verify_image(row):
        info=row['images'][0]
        assert sha(rebuild/info['path'])==info['sha256'],row['id']
    with concurrent.futures.ThreadPoolExecutor(12) as ex:
        for i,_ in enumerate(ex.map(verify_image,all_rows)):
            if (i+1)%20000==0:print(f'Rebuild image checksums verified: {i+1}',flush=True)
    repro={'all_equal':True,'image_files_verified':len(all_rows),'parts':evidence,
           'scope':'independent recompilation/rendering of every family using frozen source inputs in the pinned environment'}
    (directory/'reproducibility.json').write_text(json.dumps(repro,indent=2)+'\n')
    for s in build.SPLITS:assert sha(directory/f'{s}.jsonl')==report['splits'][s]['jsonl_sha256']
    (directory/'shards').mkdir(exist_ok=True)
    tasks=[(directory,s,j//1000,values[j:j+1000]) for s,values in rows.items() for j in range(0,len(values),1000)]
    with concurrent.futures.ThreadPoolExecutor(6) as ex:
        shards=list(ex.map(write_shard,tasks))
    (directory/'shards/index.json').write_text(json.dumps({'format':'WebDataset image+JSON tar shards','shards':shards},indent=2)+'\n')
    shutil.copytree(ROOT/'dataset/vision',directory/'dataset/vision',dirs_exist_ok=True)
    for folder,names in {
        'scripts':['download_vision_sources.py','prepare_vision_photos.py','build_vision_pool.py','finalize_vision_pool.py',
                   'snapshot_vision_sources.py','package_vision_pool.py','vision_request.py'],
        'tests':['test_vision_dataset.py'],
        'openjev':[p.name for p in (ROOT/'openjev').glob('*.py')],
    }.items():
        (directory/folder).mkdir(exist_ok=True)
        for name in names:shutil.copy2(ROOT/folder/name,directory/folder/name)
    for name in ('LICENSE','pyproject.toml'):shutil.copy2(ROOT/name,directory/name)
    shutil.copy2(ROOT/'dataset/vision/LICENSE-DATA.md',directory/'LICENSE-DATA.md')
    (directory/'README.md').write_text('''---
language: [en]
license: other
license_name: mixed-permissive-apache-ccby-cc0
license_link: LICENSE-DATA.md
task_categories: [visual-question-answering]
size_categories: [100K<n<1M]
---
# OpenJev vision pool v1

100,000 images / 200,000 finite decision slots. 85,000 training, 5,000 calibration,
and 10,000 locked evaluation images. The source mixture is 25K rendered games,
30K structured visuals, 25K CLEVR scenes, and 20K individually license-checked
Open Images photographs. No Gemini or paid generation API is used.

Canonical records: `train.jsonl`, `calibration.jsonl`, `evaluation.jsonl`.
Each record points to one local image under `images/` and records its SHA-256.
`shards/index.json` lists split-specific 1,000-record tar shards and checksums.
Each shard contains `<record-id>.png` or `.jpg` and `<record-id>.json`; the JSON
image path is adjusted to the member filename so an extracted shard is usable.

To use only shards, extract each split's shards and read their JSON files. To
reconstruct the canonical image tree, use the original manifests to copy each
extracted `<record-id>` image to the corresponding `images/...` path. Shards and
canonical files contain the same image bytes and labels.

Only `state`, `questions`, and image bytes are model inputs. Targets and
provenance are supervision. `scripts/vision_request.py` exports normal,
blank-image, mismatched-image, and compressed-image API requests without sending
them. Actual model training/inference has not been performed by this data build.

Read [methodology](dataset/vision/METHODOLOGY.md) and [license notes](LICENSE-DATA.md).
Preserve the per-photo author/license/source metadata when redistributing images.
Do not use the evaluation set or its parent states to train or calibrate.
Upstream labels can be noisy, and source snapshots can overlap base-model
pretraining; the audits do not establish model performance or contamination-free
external benchmark status.

The metadata archive includes source snapshots, code, tests, licenses, manifests,
and audit reports. It excludes image bytes and tar shards, which are distributed
separately. Follow the methodology to restore offline source inputs from a full
release and reproduce the pool. Install the local package plus
`dataset/vision/requirements-build.txt`; tokenizer auditing additionally requires
the cached tokenizer/processor named in the audit report.
''')
    files=sorted(p for p in directory.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts
                 and 'parts' not in p.relative_to(directory).parts)
    # Only referenced image files enter the release checksum manifest.
    image_paths={directory/r['images'][0]['path'] for r in all_rows}
    files=[p for p in files if 'images' not in p.relative_to(directory).parts or p in image_paths]
    (directory/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(directory)}\n' for p in files))
    archive=directory.parent/'openjev-vision-metadata.tar.gz'
    metadata=[p for p in files if 'images' not in p.relative_to(directory).parts and p.suffix!='.tar']+[directory/'SHA256SUMS']
    with archive.open('wb') as output,gzip.GzipFile(fileobj=output,mode='wb',filename='',mtime=0,compresslevel=3) as compressed:
        with tarfile.open(fileobj=compressed,mode='w') as tar:
            for path in sorted(metadata):tar_bytes(tar,'openjev-vision-v1/'+str(path.relative_to(directory)),path.read_bytes())
    summary={'images':100000,'shards':len(shards),'shard_bytes':sum(s['bytes'] for s in shards),
             'metadata_archive':str(archive),'metadata_bytes':archive.stat().st_size,'metadata_sha256':sha(archive)}
    (directory.parent/'release-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=build.RELEASE)
    p.add_argument('--rebuild',type=Path,required=True);args=p.parse_args();package(args.dataset,args.rebuild)
