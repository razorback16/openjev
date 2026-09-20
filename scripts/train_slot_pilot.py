"""Decoder-only rank-16 LoRA, one-pass noisy slots, exact allowed-label CE.

Uses frozen BF16 DiffusionGemma; encoder remains frozen and unadapted. This
custom adapter is NOT a PEFT/vLLM adapter and must not be silently merged into
the tied encoder. Artifacts include the exact loading code and base revision.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time
import torch
from torch import nn
try:
    from dotenv import load_dotenv
    load_dotenv('.env')
except ImportError:
    pass
if os.environ.get('HUGGINGFACE_API_KEY'):
    os.environ.setdefault('HF_TOKEN',os.environ['HUGGINGFACE_API_KEY'])
from transformers import DiffusionGemmaForBlockDiffusion

BASE='google/diffusiongemma-26B-A4B-it'
REV='f7f5b7f5fa82ffc52addd066915886d497f5517b'

class LoRA(nn.Module):
    def __init__(self,base,rank=16):
        super().__init__();self.base=base
        self.A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A,a=math.sqrt(5));self.scale=2.
    def forward(self,x):
        delta=torch.nn.functional.linear(torch.nn.functional.linear(x.float(),self.A),self.B)
        return self.base(x)+(delta*self.scale).to(x.dtype)

def attach(model):
    names=[]
    for name,module in list(model.model.decoder.named_modules()):
        if isinstance(module,nn.Linear) and name.endswith(('q_proj','v_proj')):
            parent,_,leaf=name.rpartition('.')
            setattr(model.model.decoder.get_submodule(parent),leaf,LoRA(module));names.append(name)
    assert names
    return names

def read(path):return [json.loads(s) for s in Path(path).open()]

def logits_for(model,row,seed):
    rng=random.Random(seed)
    canvas=row['template']+[106]
    canvas += [0]*(row['canvas_width']-len(canvas))
    for slot in row['slots']:canvas[slot['pos']]=rng.randrange(262144)
    with torch.no_grad():
        encoded=model.model.encoder(input_ids=torch.tensor([row['input_ids']],device='cuda'))
    hidden=model.model.decoder(decoder_input_ids=torch.tensor([canvas],device='cuda'),
                               past_key_values=encoded.past_key_values).last_hidden_state
    outputs=[]
    for slot in row['slots']:
        weights=model.lm_head.weight[slot['label_ids']]
        logits=torch.nn.functional.linear(hidden[0,slot['pos']],weights).float()
        outputs.append(30*torch.tanh(logits/30))
    return outputs

def metrics(rows):
    n=len(rows);correct=0;nll=0.;brier=0.;bins=[[] for _ in range(10)]
    for row in rows:
        p=torch.tensor(row['logits']).softmax(-1);y=row['target'];pred=int(p.argmax())
        correct+=pred==y;nll-=float(p[y].clamp_min(1e-12).log())
        one=torch.zeros_like(p);one[y]=1;brier+=float(((p-one)**2).sum())
        conf=float(p.max());bins[min(9,int(conf*10))].append((conf,pred==y))
    ece=sum(abs(sum(p-c for p,c in b)) for b in bins)/n
    return {'slots':n,'accuracy':correct/n,'nll':nll/n,'brier':brier/n,'ece_10_bins':ece}

def evaluate(model,data,out,name,seeds=(19,73)):
    model.eval();rows=[];start=time.time()
    with torch.no_grad():
        for i,r in enumerate(data):
            for seed in seeds:
                logits=logits_for(model,r,seed)
                for j,z in enumerate(logits):rows.append({'id':r['id'],'family':r['family'],'type':r['types'][j],
                    'slot':j,'seed':seed,'logits':z.cpu().tolist(),'target':r['targets'][j]})
            if (i+1)%16==0:print(name,i+1,'seconds',round(time.time()-start),flush=True)
    (out/f'{name}-logits.json').write_text(json.dumps(rows))
    result={'overall':metrics(rows),'families':{f:metrics([r for r in rows if r['family']==f]) for f in sorted({r['family'] for r in rows})}}
    (out/f'{name}-metrics.json').write_text(json.dumps(result,indent=2));print(name,result,flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='/workspace/pilot');p.add_argument('--steps',type=int,default=128)
    p.add_argument('--seconds',type=int,default=4200)
    p.add_argument('--train-seed',type=int,default=818)
    p.add_argument('--output-name',default='outputs')
    p.add_argument('--generation-data',type=Path)
    p.add_argument('--thinking-data',type=Path)
    p.add_argument('--replay-fraction',type=float,default=.2)
    p.add_argument('--thinking-fraction',type=float,default=.25)
    args=p.parse_args()
    from generation_replay import generation_loss,evaluate_generation,is_replay
    for fraction in [args.replay_fraction,args.thinking_fraction]:is_replay(0,fraction)
    root=Path(args.data);out=root/args.output_name;out.mkdir(exist_ok=True)
    torch.manual_seed(args.train_seed);random.seed(args.train_seed);torch.set_num_threads(8)
    start=time.time();print('LOADING',BASE,REV,flush=True)
    model=DiffusionGemmaForBlockDiffusion.from_pretrained(BASE,revision=REV,dtype=torch.bfloat16,
                    device_map={'':'cuda'},attn_implementation='sdpa',experts_implementation='grouped_mm')
    model.requires_grad_(False);names=attach(model)
    params=[p for p in model.parameters() if p.requires_grad]
    config={'base':BASE,'revision':REV,'rank':16,'alpha':32,'decoder_modules':names,'seed':args.train_seed,
            'trainable_parameters':sum(p.numel() for p in params),'torch':torch.__version__,
            'gpu':torch.cuda.get_device_name(),'objective':'allowed-label cross entropy at all noisy answer slots',
            'encoder':'frozen and unadapted','max_steps':args.steps}
    (out/'adapter_config.json').write_text(json.dumps(config,indent=2));print(config,flush=True)
    train=read(root/'train.jsonl');val=read(root/'validation.jsonl');cal=read(root/'calibration.jsonl')
    replay=read(args.generation_data/'train.jsonl') if args.generation_data else []
    replay_val=read(args.generation_data/'validation.jsonl') if args.generation_data else []
    thinking=read(args.thinking_data/'train.jsonl') if args.thinking_data else []
    thinking_val=read(args.thinking_data/'validation.jsonl') if args.thinking_data else []
    if args.generation_data and (not replay or not replay_val):raise ValueError('generation splits cannot be empty')
    if args.thinking_data and (not thinking or not thinking_val):raise ValueError('thinking splits cannot be empty')
    train_groups={r['group_id'] for r in train+thinking+replay}
    assert not train_groups & {r['group_id'] for r in val+cal+thinking_val+replay_val}
    assert not {r['id'] for r in train+thinking+replay} & {r['id'] for r in val+cal+thinking_val+replay_val}
    config.update({'generation_replay_fraction':args.replay_fraction if replay else 0,
                   'thinking_fraction_of_classification':args.thinking_fraction if thinking else 0,
                   'retention_objective':'same adapter and model for all modes; full-vocabulary denoising rehearsal'})
    (out/'adapter_config.json').write_text(json.dumps(config,indent=2))
    baseline=evaluate(model,val,out,'baseline')
    evaluate(model,cal,out,'baseline-calibration',seeds=(19,))
    if replay_val:evaluate_generation(model,replay_val,out,'baseline')
    if thinking_val:evaluate(model,thinking_val,out,'baseline-thinking')
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
    model.train();history=[];step=0;ci=gi=ti=0
    while step<args.steps and time.time()-start<args.seconds:
        optimizer.zero_grad(set_to_none=True);losses=[]
        for micro in range(2):
            index=step*2+micro;seed=args.train_seed+index
            if replay and is_replay(index,args.replay_fraction):
                row=replay[gi%len(replay)];gi+=1
                loss=generation_loss(model,row,seed)
            else:
                use_thinking=bool(thinking) and is_replay(ci,args.thinking_fraction)
                row=thinking[ti%len(thinking)] if use_thinking else train[ci%len(train)]
                ci+=1;ti+=int(use_thinking)
                logits=logits_for(model,row,seed)
                loss=torch.stack([torch.nn.functional.cross_entropy(z[None,:],torch.tensor([y],device='cuda')) for z,y in zip(logits,row['targets'])]).mean()
                if use_thinking:
                    # Retain the base's reasoning continuation as well as supervising the final slots.
                    loss=loss+.1*generation_loss(model,row['thought_replay'],seed)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            (loss/2).backward();losses.append(float(loss.detach()))
        grad=float(torch.nn.utils.clip_grad_norm_(params,1.));optimizer.step();step+=1
        entry={'step':step,'loss':sum(losses)/2,'grad_norm':grad,'elapsed_seconds':time.time()-start}
        history.append(entry);print(json.dumps(entry),flush=True)
        if step%16==0:
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items() if k.endswith(('.A','.B'))},out/'adapter.pt')
            (out/'training-history.json').write_text(json.dumps(history,indent=2))
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items() if k.endswith(('.A','.B'))},out/'adapter.pt')
    (out/'training-history.json').write_text(json.dumps(history,indent=2))
    evaluate(model,val,out,'adapted');evaluate(model,cal,out,'calibration',seeds=(19,))
    if replay_val:evaluate_generation(model,replay_val,out,'adapted')
    if thinking_val:evaluate(model,thinking_val,out,'adapted-thinking')
    # Round-trip the artifact, rather than treating a successful save as sufficient.
    model.eval()
    with torch.no_grad():
        expected=[z.cpu().clone() for z in logits_for(model,val[0],19)]
        for param in params:param.zero_()
        loaded=torch.load(out/'adapter.pt',map_location='cuda',weights_only=True)
        result=model.load_state_dict(loaded,strict=False)
        assert not result.unexpected_keys
        assert not any(k.endswith(('.A','.B')) for k in result.missing_keys)
        actual=[z.cpu() for z in logits_for(model,val[0],19)]
        for a,b in zip(expected,actual):torch.testing.assert_close(a,b,rtol=0,atol=0)
    config.update({'completed_steps':step,'elapsed_seconds':time.time()-start,
                   'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'adapter_reload_verified':True,
                   'generation_examples_seen':gi,'thinking_examples_seen':ti,'classification_examples_seen':ci})
    (out/'run.json').write_text(json.dumps(config,indent=2));print('TRAIN_DONE',flush=True)

if __name__=='__main__':main()
