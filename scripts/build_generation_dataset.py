"""Bounded, pinned NVIDIA Nemotron rehearsal subset in native DiffusionGemma format."""
import argparse
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import requests
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

CHAT=('nvidia/Nemotron-SFT-Instruction-Following-Chat-v3','be3b3e04ef605ac9d3f8f35b9d5a632f4a3a3402')
AGENT=('nvidia/Nemotron-SFT-Agentic-v2','7c804833427f633ccd53b582dbf02525fd680f78')
SOURCES={'chat':(*CHAT,'chat'),'instruction':(*CHAT,'instruction_following'),
         'tools':(*AGENT,'tool_calling'),'interactive':(*AGENT,'interactive_agent'),'search':(*AGENT,'search')}
MODEL='google/diffusiongemma-26B-A4B-it'
MODEL_REV='f7f5b7f5fa82ffc52addd066915886d497f5517b'

def normalize(row):
    messages=copy.deepcopy(row['messages']);tools=copy.deepcopy(row.get('tools') or [])
    if isinstance(tools,str):tools=json.loads(tools)
    for m in messages:
        if m.get('content') is None:
            if m.get('tool_calls'):m['content']=''
            else:raise ValueError('withheld_content')
        if not isinstance(m['content'],str):raise ValueError('unsupported_content')
        for call in m.get('tool_calls') or []:
            args=call['function'].get('arguments',{})
            if isinstance(args,str):args=json.loads(args)
            if not isinstance(args,dict):raise ValueError('non_object_tool_arguments')
            call['function']['arguments']=args
    return messages,tools

def compile_turn(tok,messages,tools,index,thinking):
    messages=copy.deepcopy(messages[:index+1])
    for m in messages:
        if not thinking:
            m.pop('reasoning_content',None);m.pop('reasoning',None)
    target=messages[-1]
    if target['role']!='assistant':raise ValueError('target_not_assistant')
    kwargs={'tools':tools or None,'enable_thinking':thinking}
    prefix=tok.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True,**kwargs)
    complete=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,**kwargs)
    if not complete.startswith(prefix):raise ValueError('template_prefix_mismatch')
    response=complete[len(prefix):]
    if not thinking:
        response='<|channel>thought\n<channel|>'+response
    elif not (target.get('reasoning_content') or target.get('reasoning')):
        raise ValueError('missing_reasoning')
    return tok.encode(prefix,add_special_tokens=False),tok.encode(response,add_special_tokens=False)

def fetch_sample(repo,rev,file,path,limit):
    if path.exists():return
    url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/data/{file}.jsonl'
    with requests.get(url,headers={'Range':f'bytes=0-{limit-1}'},stream=True,timeout=90) as r:
        r.raise_for_status();data=bytearray()
        for chunk in r.iter_content(65536):
            data.extend(chunk)
            if len(data)>=limit:break
    data=data[:limit];end=data.rfind(b'\n')
    if end<0:raise ValueError('No complete rows in bounded sample')
    path.write_bytes(data[:end+1])

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('data/openjev-generation/nemotron'))
    p.add_argument('--bytes-per-source',type=int,default=4*1024*1024)
    p.add_argument('--max-prompt',type=int,default=8192);p.add_argument('--max-response',type=int,default=4096)
    args=p.parse_args();source=args.output/'source';release=args.output/'release'
    source.mkdir(parents=True,exist_ok=True);release.mkdir(parents=True,exist_ok=True)
    load_dotenv('.env');token=os.getenv('HUGGINGFACE_API_KEY') or os.getenv('HF_TOKEN')
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_REV,token=token)
    notices=Path('dataset/generation/notices');notices.mkdir(parents=True,exist_ok=True)
    for repo,rev in [CHAT,AGENT]:
        path=hf_hub_download(repo,'README.md',repo_type='dataset',revision=rev,token=token)
        (notices/(repo.split('/')[-1]+'.md')).write_bytes(Path(path).read_bytes())
    (notices/'diffusiongemma-chat_template.jinja').write_text(tok.chat_template)
    pools={'train':[],'validation':[]};reject=collections.Counter();counts=collections.Counter();locks=[];seen=set()
    for family,(repo,rev,file) in SOURCES.items():
        path=source/f'{family}.jsonl';fetch_sample(repo,rev,file,path,args.bytes_per_source)
        locks.append({'repo':repo,'revision':rev,'file':f'data/{file}.jsonl','sampling':'complete lines in first bounded bytes',
                      'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        for line_no,line in enumerate(path.open(),1):
            row=json.loads(line)
            try:messages,tools=normalize(row)
            except (ValueError,KeyError,TypeError):reject[family+':invalid_or_withheld']+=1;continue
            metadata=row.get('metadata') or {}
            # Fail closed on explicitly restricted/unknown per-row licenses.
            license_value=row.get('license') or metadata.get('license')
            if license_value and str(license_value).lower() not in ['cc-by-4.0','apache-2.0','mit','odc-by','odc-by-1.0']:
                reject[family+':license']+=1;continue
            first=next((m['content'] for m in messages if m['role']=='user'),None)
            if not first:reject[family+':no_user']+=1;continue
            gid=hashlib.sha256(first.strip().casefold().encode()).hexdigest()
            if gid in seen:reject[family+':duplicate_prompt']+=1;continue
            seen.add(gid);split='validation' if int(gid[:8],16)%10==0 else 'train'
            turns=[i for i,m in enumerate(messages) if m['role']=='assistant']
            if family in ['chat','instruction']:
                turns=turns[-1:]  # NVIDIA explicitly permits only the last assistant turn for chat.
            else:
                tool_turns=[i for i in turns if messages[i].get('tool_calls')]
                turns=sorted(set((tool_turns[:1] if tool_turns else [])+turns[-1:]))
            for index in turns:
                for thinking in [False,True]:
                    try:ids,response=compile_turn(tok,messages,tools,index,thinking)
                    except (ValueError,TypeError,KeyError) as e:
                        reject[family+':'+str(e)[:60]]+=1;continue
                    if len(ids)>args.max_prompt or not 2<=len(response)<=args.max_response:
                        reject[family+':length']+=1;continue
                    rid=hashlib.sha256(f'{repo}:{rev}:{line_no}:{index}:{thinking}'.encode()).hexdigest()
                    record={'id':rid,'group_id':gid,'split':split,'mode':'generation','family':family,
                        'thinking':thinking,'input_ids':ids,'response_ids':response,'canvas_length':256,
                        'messages':messages[:index+1],'tools':tools,'target_turn':index,
                        'provenance':{'source':repo,'revision':rev,'source_ref':f'data/{file}.jsonl:{line_no}',
                            'license':license_value or ('CC-BY-4.0 AND ODC-BY-1.0' if family in ['chat','instruction'] else 'CC-BY-4.0'),
                            'seed_source':metadata.get('seed_dataset') or metadata.get('source'),
                            'teacher':metadata.get('model') or metadata.get('sdg_model') or row.get('model'),
                            'reasoning_origin':'NVIDIA released teacher trace' if thinking else 'removed for non-thinking view'}}
                    pools[split].append(record);counts[f'{split}:{family}:{"thinking" if thinking else "nonthinking"}']+=1
    reports={}
    for split,rows in pools.items():
        random.Random(20260920).shuffle(rows);path=release/f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        reports[split]={'records':len(rows),'groups':len({r['group_id'] for r in rows}),
                        'response_tokens':sum(len(r['response_ids']) for r in rows),
                        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    assert not {r['group_id'] for r in pools['train']}&{r['group_id'] for r in pools['validation']}
    report={'splits':reports,'counts':dict(counts),'rejected':dict(reject),'sources':locks,'seed':20260920,
            'tokenizer':MODEL,'tokenizer_revision':MODEL_REV,'standard_system_prompt':None,
            'template_sha256':hashlib.sha256(tok.chat_template.encode()).hexdigest(),
            'limits':{'max_prompt':args.max_prompt,'max_response':args.max_response},
            'limitations':'Bounded prefix pilot, not representative corpus sampling. Tool outcomes are supplied context, never executed. No retention quality claim.'}
    (release/'report.json').write_text(json.dumps(report,indent=2));Path('dataset/generation/report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
