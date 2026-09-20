"""Original rule-grounded workflow states: no teacher API content used."""
import hashlib
import json
from pathlib import Path
import random

def main():
    root=Path('data/openjev-training-pilot/procedural');root.mkdir(parents=True,exist_ok=True)
    signals={'billing':['invoice_total_mismatch','duplicate_charge','refund_not_received'],
             'authentication':['expired_session','invalid_reset_link','login_rejected'],
             'database':['replica_lag','query_timeout','deadlock'],
             'frontend':['layout_overflow','button_unresponsive','missing_styles']}
    rng=random.Random(2991)
    for split,n in [('train',96),('validation',24),('calibration',16)]:
        rows=[]
        for i in range(n):
            service=rng.choice(list(signals));affected=rng.randrange(101);loss=rng.random()<.15
            severity=4 if loss else 3 if affected>=50 else 2 if affected>=10 else 1 if affected>=1 else 0
            # Explicit policy gives independently reproducible labels, including routing.
            state=json.dumps({'event_id':f'{split}-{i}','signal':rng.choice(signals[service]),
                'affected_percent':affected,'data_loss':loss,'duration_minutes':rng.randrange(1,241),
                'queue_depth':rng.randrange(500),'region':rng.choice(['east','west','central']),
                'policy':{'owners':signals,'critical_if_data_loss':True}})
            order=list(signals);rng.shuffle(order)
            rows.append({'id':hashlib.sha256(state.encode()).hexdigest(),'group_id':f'workflow-{split}-{i}',
                'split':split,'state':state,'questions':{
                'q1':{'type':'choice','instructions':'Route the signal according to policy.owners.', 'criteria':{s:s for s in order}},
                'q2':{'type':'score','instructions':'Use priority: data_loss =>4; affected_percent>=50 =>3; >=10 =>2; >=1 =>1; otherwise 0.',
                      'criteria':['negligible','low','moderate','high','critical']},
                'q3':{'type':'noul','instructions':'Is data_loss true?'}},
                'targets':{'q1':service,'q2':str(severity),'q3':'yes' if loss else 'no'},
                'provenance':{'family':'procedural_native','task':'policy_routing','license':'Apache-2.0',
                              'generator_seed':2991,'label_kind':'deterministic rules'}})
        (root/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))

if __name__=='__main__':main()
