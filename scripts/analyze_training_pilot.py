"""Paired held-out analysis; calibration temperature fitted only on calibration."""
import json
from pathlib import Path
import numpy as np
import argparse

ROOT=Path('data/openjev-training-pilot/outputs')

def probs(z,t=1.):
    z=np.asarray(z)/t;z=z-z.max();p=np.exp(z);return p/p.sum()

def metrics(rows,t=1.):
    acc=[];nll=[];brier=[];bins=[[] for _ in range(10)]
    for r in rows:
        p=probs(r['logits'],t);y=r['target'];a=int(p.argmax())==y;c=float(p.max())
        acc.append(a);nll.append(-np.log(max(p[y],1e-12)))
        one=np.zeros(len(p));one[y]=1;brier.append(float(((p-one)**2).sum()))
        bins[min(9,int(c*10))].append((c,float(a)))
    ids=sorted({r['id'] for r in rows});disagreements=[]
    for rid in ids:
        selected=[r for r in rows if r['id']==rid]
        for slot in {r['slot'] for r in selected}:
            votes=[int(np.argmax(r['logits'])) for r in selected if r['slot']==slot]
            disagreements.append(len(set(votes))>1)
    return {'accuracy':float(np.mean(acc)),'nll':float(np.mean(nll)),'brier':float(np.mean(brier)),
            'ece_10_bins':sum(abs(sum(c-a for c,a in b)) for b in bins)/len(rows),
            'two_seed_prediction_disagreement':float(np.mean(disagreements)),
            'records':len(ids),'seed_slot_observations':len(rows)}

def main():
    global ROOT
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args();ROOT=args.root
    base=json.loads((ROOT/'baseline-logits.json').read_text())
    adapted=json.loads((ROOT/'adapted-logits.json').read_text())
    cal=json.loads((ROOT/'calibration-logits.json').read_text())
    key=lambda r:(r['id'],r['slot'],r['seed'])
    base=sorted(base,key=key);adapted=sorted(adapted,key=key)
    assert [key(r) for r in base]==[key(r) for r in adapted]
    assert not {r['id'] for r in cal}&{r['id'] for r in adapted}
    temperatures=np.exp(np.linspace(np.log(.2),np.log(5),401))
    temp=float(min(temperatures,key=lambda t:metrics(cal,t)['nll']))
    delta={}
    for a,b in zip(base,adapted):
        assert a['target']==b['target']
        delta.setdefault(a['id'],[]).append(float(np.argmax(b['logits'])==b['target'])-float(np.argmax(a['logits'])==a['target']))
    # Cluster bootstrap by record, retaining all seeds/slots of each sampled record.
    ds=list(delta.values());sums=np.array([sum(v) for v in ds]);counts=np.array([len(v) for v in ds])
    rng=np.random.default_rng(818);ix=rng.integers(0,len(ds),size=(5000,len(ds)))
    boots=sums[ix].sum(1)/counts[ix].sum(1)
    report={'baseline':metrics(base),'adapted':metrics(adapted),'adapted_calibrated':metrics(adapted,temp),
            'temperature':temp,'accuracy_delta_95pct_record_bootstrap':np.quantile(boots,[.025,.975]).tolist(),
            'family_metrics':{f:{'baseline':metrics([r for r in base if r['family']==f]),
                                 'adapted':metrics([r for r in adapted if r['family']==f])} for f in sorted({r['family'] for r in base})},
            'limitations':['One training seed and small validation sample; no production-quality claim.',
              'BF16 Transformers one-pass evaluation; no vLLM/NVFP4 deployment-equivalence claim.',
              'Decoder-only custom LoRA; encoder frozen and unadapted; not directly deployable through current vLLM.',
              'No image training or image evaluation in this text pilot.',
              'Seed disagreement is measured; actual engine reread rate was not measured.',
              'Temperature fitted globally on a small separate calibration set; ECE is noisy.']}
    base_cal_file=ROOT/'baseline-calibration-logits.json'
    if base_cal_file.exists():
        base_cal=json.loads(base_cal_file.read_text())
        assert not {r['id'] for r in base_cal}&{r['id'] for r in base}
        base_temp=float(min(temperatures,key=lambda t:metrics(base_cal,t)['nll']))
        report['baseline_temperature']=base_temp
        report['baseline_calibrated']=metrics(base,base_temp)
        nll_delta={}
        for a,b in zip(base,adapted):
            pa=probs(a['logits'],base_temp)[a['target']]
            pb=probs(b['logits'],temp)[b['target']]
            nll_delta.setdefault(a['id'],[]).append(float(np.log(max(pa,1e-12))-np.log(max(pb,1e-12))))
        ns=np.array([sum(nll_delta[k]) for k in delta])
        report['calibrated_nll_delta_95pct_record_bootstrap']=np.quantile(ns[ix].sum(1)/counts[ix].sum(1),[.025,.975]).tolist()
    source_file=ROOT.parent/'sources.jsonl'
    if source_file.exists():
        sources={r['id']:r for r in map(json.loads,source_file.open())}
        report['games_acceptable_move_accuracy']={}
        for name,rows in [('baseline',base),('adapted',adapted)]:
            games=[r for r in rows if r['family']=='games']
            correct=[int(np.argmax(r['logits'])) in sources[r['id']]['acceptable_indices'][str(r['slot'])] for r in games]
            report['games_acceptable_move_accuracy'][name]=float(np.mean(correct))
    (ROOT/'analysis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
