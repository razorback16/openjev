"""Convert a local image record to an OpenJev request, including ablation controls."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

from PIL import Image


def request_for(row,root,arm='normal',alternate=None):
    info=row['images'][0];path=root/info['path'];data=path.read_bytes();ctype=info['content_type']
    if hashlib.sha256(data).hexdigest()!=info['sha256']:raise ValueError('Image checksum mismatch')
    if arm=='mismatch':
        if alternate is None or alternate['id']==row['id']:raise ValueError('A different record is required')
        other=alternate['images'][0];data=(root/other['path']).read_bytes();ctype=other['content_type']
        if hashlib.sha256(data).hexdigest()!=other['sha256']:raise ValueError('Alternate checksum mismatch')
    elif arm in ('blank','compressed'):
        image=Image.new('RGB',(info['width'],info['height']),'#808080') if arm=='blank' else Image.open(io.BytesIO(data)).convert('RGB')
        if arm=='compressed':image.thumbnail((384,384),Image.Resampling.LANCZOS)
        buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=45 if arm=='compressed' else 90)
        data=buffer.getvalue();ctype='image/jpeg'
    elif arm!='normal':raise ValueError(arm)
    # Only model inputs enter the API payload; no targets, annotations or provenance.
    return {'model':'openjev-latest','state':row['state'],'questions':row['questions'],
            'images':[{'content_type':ctype,'base64':base64.b64encode(data).decode()}]}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=Path('data/openjev-vision/release'))
    p.add_argument('--index',type=int,default=0);p.add_argument('--arm',choices=['normal','blank','mismatch','compressed'],default='normal')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    rows=[json.loads(line) for line in (args.dataset/'evaluation.jsonl').open()]
    row=rows[args.index];other=next(r for r in rows if r['id']!=row['id'] and r['provenance']['family']==row['provenance']['family'])
    args.output.write_text(json.dumps(request_for(row,args.dataset,args.arm,other))+'\n')
