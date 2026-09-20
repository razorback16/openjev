"""Join, audit and freeze the 100K image pool; never train a model."""
import argparse
import asyncio
import collections
import concurrent.futures
import hashlib
import gzip
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

import build_vision_pool as build

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def image_check(item):
    directory,row=item;info=row['images'][0];path=directory/info['path']
    assert path.stat().st_size<=5*1024*1024
    checksum=sha(path);assert checksum==info['sha256']
    with Image.open(path) as image:
        image.load();assert image.size==(info['width'],info['height'])
        rgb=image.convert('RGB');pixels=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
        small=np.asarray(rgb.resize((9,8)).convert('L'),dtype=np.int16)
        dhash=sum(int(bit)<<i for i,bit in enumerate((small[:,1:]>small[:,:-1]).flatten()))
        thumb=np.asarray(rgb.resize((32,32)),dtype=np.int16)
        assert np.std(thumb)>3,'Blank/corrupt image'
    return row['id'],checksum,pixels,dhash,thumb,path.stat().st_size


def check_labels(row,parents):
    p=row['provenance'];family=p['family'];t=row['targets'];q=row['questions']
    if family=='games':
        parent=parents[p['parent_record_id']]
        assert row['group_id']==parent['group_id']
        expected_split='evaluation' if parent['split']=='validation' else parent['split']
        assert row['split']==expected_split
        if p['game']=='chess':
            assert p['move_mapping'][t['q1']]==parent['targets']['q1']
            assert '/' not in row['state']
            assert all('#' not in v and '+' not in v for v in q['q1']['criteria'].values())
        else:assert t==parent['targets']
        if p['game']=='texas_holdem':assert 'Your hole cards:' not in row['state']
    elif family=='structured':
        kind=p['kind']
        if kind in ('bar','line'):
            assert t['q1']==p['labels'][p['values'].index(max(p['values']))]
            truth=p['values'][p['probe_index']]>p['threshold']
        elif kind=='table':
            changes=[b-a for a,b in zip(p['starts'],p['ends'])]
            assert t['q1']==p['labels'][changes.index(max(changes))]
            truth=changes[p['probe_index']]>0
        elif kind=='invoice':
            sub=sum(a*b for a,b in zip(p['qty'],p['unit_cents']));total=sub+sub//10
            assert t['q1']==str(total)==str(p['total_cents']);truth=total>p['threshold_cents']
        else:
            assert t['q1']=='ABCD'[p['actions'].index(p['target_action'])];truth=p['status']=='Failed'
        assert t['q2']==('yes' if truth else 'no')
    elif family=='clevr':
        assert (p['original_split']=='val')==(row['split']=='evaluation')
        for i,source in enumerate(p['source_questions']):
            assert build.clevr_execute(source['program'],p['scene'])==t[f'q{i+1}']==str(source['answer'])
    else:
        assert p['license'].startswith('CC-BY-') or p['license']=='CC0-1.0'
        assert p['license_evidence']['@type']=='ImageObject'
        assert p['license_evidence']['acquireLicensePage'].rstrip('/').split('/')[-1]==p['original_metadata']['OriginalLandingURL'].rstrip('/').split('/')[-1]
        for i,(label,target) in enumerate(p['selected_labels']):
            assert t[f'q{i+1}']==target
            assert label in p['all_verified_labels']['positive' if target=='yes' else 'negative']
        assert (p['original_split']=='train')==(row['split']=='train')


async def schema_audit(rows,directory):
    from openjev.engine import Engine
    from openjev.config import Settings
    from transformers import AutoTokenizer, AutoProcessor
    name='nvidia/diffusiongemma-26B-A4B-it-NVFP4'
    tokenizer=AutoTokenizer.from_pretrained(name,local_files_only=True)
    processor=AutoProcessor.from_pretrained(name,local_files_only=True)
    vision=processor._get_num_multimodal_tokens(image_sizes=[(r['images'][0]['height'],r['images'][0]['width']) for r in rows])
    representatives={next(i for i,r in enumerate(rows) if r['provenance']['family']==f) for f in ('games','structured','clevr','photos')}
    representatives.add(max(range(len(rows)),key=lambda i:rows[i]['images'][0]['width']/rows[i]['images'][0]['height']))
    representatives.add(min(range(len(rows)),key=lambda i:rows[i]['images'][0]['width']/rows[i]['images'][0]['height']))
    for i in representatives:
        with Image.open(directory/rows[i]['images'][0]['path']) as image:
            processed=processor.image_processor(images=image.convert('RGB'),return_tensors='np')
            assert int(processed['num_soft_tokens_per_image'][0])==vision.num_image_tokens[i]
    engine=Engine(Settings(canvas=64),tokenizer);shapes=set();max_tokens=0;slots=0;max_combined=0
    try:
        for index,r in enumerate(rows):
            schema=engine.build_schema(r['questions']);qs,fmt=schema['questions'],schema['format']
            signature=tuple((q['id'],q['type'],tuple(q['labels'])) for q in qs)
            if signature not in shapes:
                for group in engine.groups(qs,fmt):engine.resolve_template(group,fmt)
                shapes.add(signature)
            for q in qs:
                assert str(r['targets'][q['key']]) in [str(n) for n,_ in q['choices']]
            slots+=len(qs)
            length=len(engine.chat_prompt_ids(engine.system_text(qs,fmt),r['state']))
            max_tokens=max(max_tokens,length)
            combined=length+vision.num_image_tokens[index]+64+8
            assert combined<=65536
            max_combined=max(max_combined,combined)
        return {'tokenizer':name,'tokenizer_sha256':hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                'records':len(rows),'decision_slots':slots,'template_shapes':len(shapes),'max_text_prompt_tokens':max_tokens,
                'image_processor':type(processor.image_processor).__name__,
                'image_processor_sha256':hashlib.sha256(processor.image_processor.to_json_string().encode()).hexdigest(),
                'max_image_tokens':max(vision.num_image_tokens),
                'actual_image_preprocessing_samples_verified':len(representatives),
                'max_estimated_combined_tokens_with_canvas_and_marker_margin':max_combined,
                'image_token_accounting':'cached multimodal processor dimensions; backend inference not run',
                'invalid_templates':0,'invalid_targets':0,'model_inference_performed':False}
    finally:await engine.close()


def finalize(directory,workers):
    rows=[json.loads(line) for family in ('games','structured','clevr','photos') for line in (directory/'parts'/f'{family}.jsonl').open()]
    assert len(rows)==100000
    assert dict(collections.Counter(r['provenance']['family'] for r in rows))=={'games':25000,'structured':30000,'clevr':25000,'photos':20000}
    assert dict(collections.Counter(r['split'] for r in rows))=={'train':85000,'calibration':5000,'evaluation':10000}
    ids,groups=set(),{}
    parents={}
    frozen=directory/'sources/games.selected.jsonl.gz'
    if frozen.exists():
        with gzip.open(frozen,'rt') as f:
            for line in f:
                r=json.loads(line);parents[r['id']]=r
    else:
        for source in ('openjev-games','openjev-holdem'):
            for split in ('train','validation','calibration'):
                for line in (ROOT/'data'/source/'release'/f'{split}.jsonl').open():
                    r=json.loads(line);parents[r['id']]=r
    for r in rows:
        assert r['id'] not in ids;ids.add(r['id'])
        assert groups.setdefault(r['group_id'],r['split'])==r['split']
        check_labels(r,parents)
    print('All labels and source split assignments verified',flush=True)
    encoded,pixels={},{};dups=[];photo_hashes=[];image_bytes=0;by_id={r['id']:r for r in rows}
    with concurrent.futures.ThreadPoolExecutor(workers) as ex:
        for i,(rid,filehash,pixelhash,dh,thumb,nbytes) in enumerate(ex.map(image_check,((directory,r) for r in rows))):
            image_bytes+=nbytes
            if pixelhash in pixels:dups.append([pixels[pixelhash],rid])
            pixels[pixelhash]=rid;encoded[filehash]=rid
            if by_id[rid]['provenance']['family']=='photos':photo_hashes.append((rid,dh,thumb))
            if (i+1)%10000==0:print(f'Decoded and hashed {i+1} images',flush=True)
    cross_duplicates=[pair for pair in dups if by_id[pair[0]]['split']!=by_id[pair[1]]['split']]
    if cross_duplicates:raise ValueError(f'Cross-split duplicate decoded images: {cross_duplicates[:10]}')
    # Conservative near-duplicate screening: close dHash plus low thumbnail RGB error.
    buckets=collections.defaultdict(list);near=[]
    for i,(rid,dh,thumb) in enumerate(photo_hashes):
        candidates=set(j for k in range(4) for j in buckets[(k,(dh>>(16*k))&65535)])
        for j in candidates:
            other,oh,ot=photo_hashes[j]
            if by_id[other]['split']==by_id[rid]['split']:continue
            if (dh^oh).bit_count()<=2 and float(np.mean(np.abs(thumb-ot)))<3:
                near.append([other,rid])
        for k in range(4):buckets[(k,(dh>>(16*k))&65535)].append(i)
    (directory/'near_duplicate_candidates.json').write_text(json.dumps(near,indent=2)+'\n')
    if near:raise ValueError(f'Cross-split near-photo duplicates require replacement: {len(near)}; see near_duplicate_candidates.json')
    print('Image integrity and cross-split duplicate screening passed',flush=True)
    token_audit=asyncio.run(schema_audit(rows,directory))
    (directory/'tokenizer_audit.json').write_text(json.dumps(token_audit,indent=2)+'\n')
    report={'version':'openjev-vision-v1','seed':build.SEED,'images':len(rows),'decision_slots':sum(len(r['questions']) for r in rows),
            'unique_encoded_image_hashes':len(encoded),'unique_decoded_image_hashes':len(pixels),'within_split_duplicate_pairs':dups,
            'cross_split_exact_duplicates':0,'cross_split_near_photo_candidates':0,'image_bytes':image_bytes,
            'versions':{p:importlib.metadata.version(p) for p in ('Pillow','numpy','transformers','tokenizers')},
            'builder_sha256':sha(ROOT/'scripts/build_vision_pool.py'),'splits':{},'model_inference_performed':False}
    for split in build.SPLITS:
        selected=sorted([r for r in rows if r['split']==split],key=lambda r:r['id'])
        path=directory/f'{split}.jsonl';path.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in selected))
        report['splits'][split]={'images':len(selected),'decision_slots':sum(len(r['questions']) for r in selected),
            'families':dict(collections.Counter(r['provenance']['family'] for r in selected)),'jsonl_sha256':sha(path)}
    evaluation=[r for r in rows if r['split']=='evaluation']
    lock={'records':10000,'evaluation_jsonl_sha256':report['splits']['evaluation']['jsonl_sha256'],
          'image_manifest_sha256':build.digest(sorted((r['id'],r['images'][0]['sha256']) for r in evaluation)),
          'rule':'Do not use evaluation records, their underlying parent states, or alternate renders for training or calibration.'}
    (directory/'evaluation.lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='within_split_duplicate_pairs'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=build.RELEASE)
    p.add_argument('--workers',type=int,default=12);args=p.parse_args();finalize(args.dataset,args.workers)
