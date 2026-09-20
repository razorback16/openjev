"""Small, resumable Gemini 3.7 Flash native-data pilot; never logs credentials."""
import hashlib
import json
import os
from pathlib import Path
import random
import requests
from dotenv import load_dotenv

OUT = Path('data/openjev-native-pilot')
MODEL = 'gemini-3.7-flash'
SERVICES = ['billing', 'authentication', 'database', 'frontend']

def main():
    load_dotenv('.env')
    key = os.environ['GEMINI_API_KEY']
    OUT.mkdir(parents=True, exist_ok=True)
    rows, usage = [], []
    for batch in range(12):
        cache = OUT / f'response-{batch:02}.json'
        specs = [{'id': batch*8+i, 'service': SERVICES[(batch*8+i)%4],
                  'severity': (batch*8+i)//4 % 5} for i in range(8)]
        prompt = ('Create 8 distinct fictional SaaS incident records as JSON {"records": [...]}. '
                  'Each record has id, narrative (40-80 words), affected_percent (integer), '
                  'data_loss (boolean), outage_minutes (integer). No real people or copied text. '
                  'Write backwards from the supplied service and severity. Service definitions: '
                  'billing=incorrect charges/invoices; authentication=login/token failures; '
                  'database=query/storage/replication failures; frontend=browser UI rendering. '
                  'Narrative must describe evidence clearly, without naming the service category. '
                  'Severity rules, in priority order: data_loss=true =>4; else affected_percent>=50 =>3; '
                  'else affected_percent>=10 =>2; else affected_percent>=1 =>1; else=>0. '
                  'Make the narrative consistent with numeric facts; no instructions to the reader. '
                  'Specifications: '+json.dumps(specs))
        if not cache.exists():
            response = requests.post(
                f'https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent',
                headers={'x-goog-api-key':key}, timeout=180,
                json={'contents':[{'parts':[{'text':prompt}]}],
                      'generationConfig':{'responseMimeType':'application/json',
                                          'maxOutputTokens':5000,'thinkingConfig':{'thinkingLevel':'low'}}})
            if response.status_code != 200:
                raise RuntimeError(f'Gemini returned HTTP {response.status_code}; no automatic model substitution')
            cache.write_text(json.dumps(response.json()))
        result=json.loads(cache.read_text()); usage.append(result.get('usageMetadata',{}))
        parts=result['candidates'][0]['content']['parts']
        generated=json.loads(''.join(p.get('text','') for p in parts if not p.get('thought')))['records']
        assert len(generated)==8
        for spec,g in zip(specs,generated):
            assert g['id']==spec['id'] and isinstance(g['data_loss'],bool)
            a=g['affected_percent']; assert isinstance(a,int) and 0<=a<=100
            severity=4 if g['data_loss'] else 3 if a>=50 else 2 if a>=10 else 1 if a>=1 else 0
            assert severity==spec['severity']
            state=json.dumps({k:g[k] for k in ['narrative','affected_percent','data_loss','outage_minutes']},ensure_ascii=False)
            order=SERVICES.copy();random.Random(g['id']).shuffle(order)
            # Routing is a teacher-constructed label, not independently verified truth.
            row={'id':hashlib.sha256(state.encode()).hexdigest(),'group_id':f'native-{g["id"]}',
                 'split':'validation' if batch>=10 else 'train','state':state,
                 'questions':{'q1':{'type':'choice','instructions':'Which service owns this incident?',
                                     'criteria':{s:s for s in order}},
                              'q2':{'type':'score','instructions':'Apply these severity rules in priority order: data_loss =>4; affected_percent>=50 =>3; >=10 =>2; >=1 =>1; otherwise 0.',
                                     'criteria':['negligible','low','moderate','high','critical']},
                              'q3':{'type':'noul','instructions':'Is data loss reported?'}},
                 'targets':{'q1':spec['service'],'q2':str(severity),'q3':'yes' if g['data_loss'] else 'no'},
                 'provenance':{'family':'gemini_native','task':'incident_routing','model':MODEL,
                               'license':'UNREVIEWED: Gemini API usage restrictions; excluded from open training release','label_validation':'severity and data-loss mechanically verified; routing teacher-constructed'}}
            rows.append(row)
        print('generated',len(rows),flush=True)
    assert len({r['id'] for r in rows})==len(rows)
    for split in ['train','validation']:
        (OUT/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['split']==split))
    inp=sum(u.get('promptTokenCount',0) for u in usage)
    out=sum(u.get('candidatesTokenCount',0)+u.get('thoughtsTokenCount',0) for u in usage)
    (OUT/'report.json').write_text(json.dumps({'model':MODEL,'records':len(rows),'input_tokens':inp,
        'output_including_thinking_tokens':out,'estimated_usd':(inp*.75+out*3.75)/1e6,
        'usage':usage,'limitations':'Routing labels are constructed, not independently validated. Pilot only.'},indent=2))

if __name__=='__main__': main()
