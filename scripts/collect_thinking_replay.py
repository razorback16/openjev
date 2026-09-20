"""Collect frozen-base thoughts through OpenJev's existing think() path.

No labels are sent to the generator. Splits inherit parent groups. This script
uses an already-running local/user-specified backend; it never provisions GPUs.
"""
import argparse
import asyncio
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from openjev.engine import Engine
from openjev.config import Settings
from transformers import AutoTokenizer

def compile_thinking(engine,row,prefix):
    schema=engine.build_schema(row['questions']);qs=schema['questions'];fmt=schema['format']
    prompt=engine.chat_prompt_ids(engine.system_text(qs,fmt),row['state'],thinking=True)
    opening=prompt+engine.thought_open
    if prefix[:len(opening)]!=opening or prefix[-len(engine.thought_close):]!=engine.thought_close:
        raise ValueError('thought prefix does not match the production prompt')
    template,slots=engine.resolve_template(qs,fmt,head=[])
    targets=[[name for name,_ in q['choices']].index(str(row['targets'][q['key']])) for q in qs]
    return {'id':row['id'],'group_id':row['group_id'],'mode':'thinking_classification',
            'family':row['provenance'].get('family'),'input_ids':prefix,'template':template,'slots':slots,
            'targets':targets,'types':[q['type'] for q in qs],'canvas_width':engine.canvas_width(template),
            'thought_replay':{'input_ids':prompt,'response_ids':prefix[len(prompt):],'canvas_length':256},
            'provenance':{'teacher':'frozen original DiffusionGemma via Engine.think',
                          'parent':row['provenance'],'thoughts_are_not_verified_reasoning_labels':True}}

async def run(args):
    tok=AutoTokenizer.from_pretrained(Settings().tokenizer,local_files_only=True)
    engine=Engine(dataclasses.replace(Settings(),upstream=args.upstream),tok)
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        for split in ['train','validation']:
            path=args.source/f'{split}.jsonl';out=args.output/f'{split}.jsonl'
            if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
            with out.open('w') as stream:
                for index,line in enumerate(path.open()):
                    if index>=args.count:break
                    row=json.loads(line);schema=engine.build_schema(row['questions'])
                    system=engine.system_text(schema['questions'],schema['format'])
                    prefix,_,_=await engine.think(system,row['state'],args.budget)
                    stream.write(json.dumps(compile_thinking(engine,row,prefix))+'\n');stream.flush()
    finally:await engine.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('data/openjev-thinking'))
    p.add_argument('--upstream',default='http://127.0.0.1:8000');p.add_argument('--count',type=int,default=128)
    p.add_argument('--budget',type=int,default=256);asyncio.run(run(p.parse_args()))
