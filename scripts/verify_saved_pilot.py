"""Reload a saved adapter in a fresh process and reproduce held-out logits."""
import json
from pathlib import Path
import torch
from train_slot_pilot import BASE,REV,attach,read,logits_for
from transformers import DiffusionGemmaForBlockDiffusion

root=Path('/workspace/pilot');out=root/'outputs';torch.set_num_threads(8)
model=DiffusionGemmaForBlockDiffusion.from_pretrained(BASE,revision=REV,dtype=torch.bfloat16,
    device_map={'':'cuda'},attn_implementation='sdpa',experts_implementation='grouped_mm')
model.requires_grad_(False);attach(model)
result=model.load_state_dict(torch.load(out/'adapter.pt',map_location='cuda',weights_only=True),strict=False)
assert not result.unexpected_keys and not any(k.endswith(('.A','.B')) for k in result.missing_keys)
expected={(r['id'],r['slot'],r['seed']):r['logits'] for r in json.loads((out/'adapted-logits.json').read_text())}
model.eval();errors=[]
with torch.no_grad():
    for row in read(root/'validation.jsonl')[:12]:
        for j,actual in enumerate(logits_for(model,row,19)):
            target=torch.tensor(expected[row['id'],j,19],device='cuda')
            torch.testing.assert_close(actual,target,rtol=0,atol=.001)
            errors.append(float((actual-target).abs().max()))
report={'records':12,'slots':len(errors),'max_absolute_logit_difference':max(errors),'passed':True}
(out/'fresh-process-reload.json').write_text(json.dumps(report,indent=2));print(report)
