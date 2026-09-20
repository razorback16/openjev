"""Freeze a small, group-disjoint pilot using the production slot compiler."""
import asyncio
import hashlib
import json
from pathlib import Path
import random
import sys
import os
from dotenv import load_dotenv
load_dotenv('.env')
if os.environ.get('HUGGINGFACE_API_KEY'):
    os.environ.setdefault('HF_TOKEN',os.environ['HUGGINGFACE_API_KEY'])
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer
from openjev.engine import Engine
from openjev.config import Settings

OUT=Path('data/openjev-training-pilot')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained('nvidia/diffusiongemma-26B-A4B-it-NVFP4',local_files_only=True)
    engine=Engine(Settings(),tok)
    seen=set(); report={}
    for split,count in [('train',512),('validation',96),('calibration',48)]:
        pool=[]
        for family,root in [('public','data/openjev-pilot/permissive-release'),('games','data/openjev-games/release'),('native','data/openjev-training-pilot/procedural')]:
            path=Path(root)/f'{split}.jsonl'
            if not path.exists():continue
            candidates=[json.loads(s) for s in path.open()]
            random.Random(381+len(split)).shuffle(candidates)
            accepted=[]
            quota=(int(count*.75) if family=='public' else int(count*.125))
            for row in candidates:
                gid=family+':'+row['group_id']
                if gid in seen:continue
                try:
                    schema=engine.build_schema(row['questions']); qs=schema['questions']
                    template,slots=engine.resolve_template(qs,schema['format'])
                    ids=engine.chat_prompt_ids(engine.system_text(qs,schema['format']),row['state'])
                    if len(ids)>1536:continue
                    labels=[]
                    for q in qs:
                        names=[p[0] for p in q['choices']]
                        labels.append(names.index(str(row['targets'][q['key']])))
                    item={'id':row['id'],'group_id':gid,'family':family,'task':row['provenance'].get('task'),
                          'input_ids':ids,'template':template,'slots':slots,'targets':labels,
                          'types':[q['type'] for q in qs],'canvas_width':engine.canvas_width(template)}
                except (ValueError,KeyError):continue
                accepted.append(item);seen.add(gid)
                if len(accepted)>=quota:break
            pool.extend(accepted)
        random.Random(818).shuffle(pool)
        path=OUT/f'{split}.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in pool))
        report[split]={'records':len(pool),'slots':sum(len(r['slots']) for r in pool),
                       'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                       'families':{f:sum(r['family']==f for r in pool) for f in ['public','games','native']}}
    (OUT/'data-report.json').write_text(json.dumps(report,indent=2))
    asyncio.run(engine.close());print(json.dumps(report,indent=2))

if __name__=='__main__':main()
