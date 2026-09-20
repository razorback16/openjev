"""Freeze 20K Open Images photos after checking original image-specific licenses."""
import collections
import concurrent.futures
import csv
import datetime
import hashlib
import io
import json
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, ImageOps
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/openjev-vision/source'
PHOTO = SOURCE/'photos'
LOCAL = threading.local()
ALLOW = {f'https://creativecommons.org/licenses/by/{v}/':f'CC-BY-{v}' for v in ('2.0','2.5','3.0','4.0')}
ALLOW['https://creativecommons.org/publicdomain/zero/1.0/'] = 'CC0-1.0'


def session():
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
        LOCAL.session.headers['User-Agent'] = 'OpenJevDataset/1.0 (https://github.com/razorback16/openjev; image license verification)'
    return LOCAL.session


def image_object(page, landing):
    target = landing.rstrip('/').split('/')[-1]
    for block in re.findall(r'<script[^>]+type=[\"\']application/ld\+json[\"\'][^>]*>(.*?)</script>', page, re.S):
        try: data = json.loads(block)
        except ValueError: continue
        if not isinstance(data, dict): continue
        for item in data.get('@graph', [data]):
            if item.get('@type') != 'ImageObject': continue
            actual = item.get('acquireLicensePage', '').rstrip('/').split('/')[-1]
            license_url = item.get('license', '').replace('http://','https://')
            if actual == target and license_url in ALLOW:
                return item, ALLOW[license_url]
    return None, None


def fetch_candidate(item):
    meta, labels, original_split = item
    image_id = meta['ImageID']; evidence_file = PHOTO/'evidence'/f'{image_id}.json'
    image_file = PHOTO/'images'/f'{image_id}.jpg'
    if evidence_file.exists():
        result = json.loads(evidence_file.read_text())
        if result.get('accepted') and image_file.exists(): return result
        if result.get('permanent_rejection'): return None
    result = None
    for attempt in range(4):
        try:
            r = session().get(meta['OriginalLandingURL'], timeout=(15,35))
            if r.status_code in (429,500,502,503,504):
                time.sleep(min(30, 2**(attempt+1))); continue
            obj, license_name = image_object(r.text, meta['OriginalLandingURL']) if r.status_code == 200 else (None,None)
            base = {'image_id':image_id, 'original_split': original_split, 'source_metadata':meta,
                    'license_checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'landing_http_status':r.status_code, 'landing_html_sha256':hashlib.sha256(r.content).hexdigest()}
            if obj is None:
                result = {**base, 'accepted':False, 'permanent_rejection':True, 'reason':'no_permissive_image_specific_license'}
                break
            url = f'https://open-images-dataset.s3.amazonaws.com/{original_split}/{image_id}.jpg'
            data = session().get(url, timeout=(15,60)); data.raise_for_status()
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(data.content))).convert('RGB')
            if min(image.size) < 128:
                result = {**base,'accepted':False,'permanent_rejection':True,'reason':'too_small'}; break
            original_size = image.size
            image.thumbnail((1024,1024), Image.Resampling.LANCZOS)
            image.save(image_file, quality=92, optimize=False)
            result = {**base,'accepted':True,'license':license_name,'license_evidence':obj,
                      'annotation_license':'CC-BY-4.0','labels':labels, 'download_url':url,
                      'download_sha256':hashlib.sha256(data.content).hexdigest(),
                      'image_sha256':hashlib.sha256(image_file.read_bytes()).hexdigest(),
                      'source_dimensions':original_size,'dimensions':image.size,
                      'modifications':'EXIF orientation applied; RGB conversion; fit within 1024px; JPEG quality 92'}
            break
        except (requests.RequestException, OSError, ValueError):
            time.sleep(min(20, 2**attempt))
    if result is not None:
        evidence_file.write_text(json.dumps(result, ensure_ascii=False, separators=(',',':'))+'\n')
    return result if result and result['accepted'] else None


def main():
    for folder in ('images','evidence'): (PHOTO/folder).mkdir(parents=True,exist_ok=True)
    classes = dict(csv.reader((SOURCE/'classes.csv').open()))
    selection, seen_md5, seen_pixels, seen_landing = [], set(), set(), set()
    perceptual=[];buckets=collections.defaultdict(list);quality=collections.Counter()
    for split, target in [('validation',3000),('train',17000)]:
        metadata = {}
        # A bounded prefix gives plenty of replacement candidates and avoids multi-GB objects.
        with (SOURCE/f'{split}-images.csv').open() as f:
            for row in csv.DictReader(f):
                if row['License'].replace('http://','https://') not in ALLOW: continue
                if 'flickr.com/photos/' not in row['OriginalLandingURL']: continue
                metadata[row['ImageID']] = row
                if len(metadata) >= (12000 if split == 'validation' else 65000): break
        labels = collections.defaultdict(lambda:{'positive':[],'negative':[]})
        with (SOURCE/f'{split}-labels.csv').open() as f:
            for row in csv.DictReader(f):
                iid = row['ImageID']
                if iid in metadata and row['LabelName'] in classes:
                    if row['Confidence'] not in ('0','1'):raise ValueError('Expected a verified binary label')
                    labels[iid]['positive' if row['Confidence']=='1' else 'negative'].append(classes[row['LabelName']])
        candidates = [(m, labels[iid], split) for iid,m in metadata.items()
                      if labels[iid]['positive'] and labels[iid]['negative']]
        print(f'{split}: {len(candidates)} annotated candidates',flush=True)
        accepted = 0
        # Fixed ordered batches make selection independent of thread completion order.
        with concurrent.futures.ThreadPoolExecutor(16) as ex:
            for start in range(0,len(candidates),64):
                for result in ex.map(fetch_candidate,candidates[start:start+64]):
                    if result is None or accepted >= target: continue
                    original = result['source_metadata']['OriginalMD5']
                    pixels = result['image_sha256']; landing = result['source_metadata']['OriginalLandingURL'].rstrip('/')
                    if original in seen_md5 or pixels in seen_pixels or landing in seen_landing: continue
                    with Image.open(PHOTO/'images'/f'{result["image_id"]}.jpg') as image:
                        rgb=image.convert('RGB');thumb=np.asarray(rgb.resize((32,32)),dtype=np.int16)
                        small=np.asarray(rgb.resize((9,8)).convert('L'),dtype=np.int16)
                    if float(np.std(thumb))<=3:
                        quality['low_contrast_rejected']+=1;continue
                    dh=sum(int(bit)<<i for i,bit in enumerate((small[:,1:]>small[:,:-1]).flatten()))
                    close=set(j for k in range(4) for j in buckets[(k,(dh>>(16*k))&65535)])
                    if any((dh^perceptual[j][0]).bit_count()<=2 and float(np.mean(np.abs(thumb-perceptual[j][1])))<3 for j in close):
                        quality['near_duplicate_rejected']+=1;continue
                    for k in range(4):buckets[(k,(dh>>(16*k))&65535)].append(len(perceptual))
                    perceptual.append((dh,thumb))
                    seen_md5.add(original);seen_pixels.add(pixels);seen_landing.add(landing)
                    result['split'] = 'train' if split == 'train' else ('evaluation' if accepted < 2000 else 'calibration')
                    selection.append(result);accepted += 1
                if (start//64) % 8 == 0: print(f'{split}: accepted {accepted}/{target}, checked {min(start+64,len(candidates))}',flush=True)
                (PHOTO/'selection.partial.jsonl').write_text(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in selection))
                if accepted == target: break
        if accepted != target: raise ValueError(f'Insufficient verified photos in {split}: {accepted}/{target}')
    (PHOTO/'selection.jsonl').write_text(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in selection))
    (PHOTO/'selection-quality-report.json').write_text(json.dumps(dict(quality),indent=2)+'\n')
    print('Photo subset complete: 20000 license-verified images',flush=True)


if __name__ == '__main__': main()
