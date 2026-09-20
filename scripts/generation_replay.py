"""Generation rehearsal for DiffusionGemma, not shifted autoregressive SFT."""
import math
import random
import torch
from torch.utils.checkpoint import checkpoint

def is_replay(index,fraction):
    if not 0<=fraction<=1:raise ValueError('replay fraction must be in [0,1]')
    return math.floor((index+1)*fraction+1e-9)>math.floor(index*fraction+1e-9)

def noisy_block(row,seed,vocab_size=262144):
    rng=random.Random(seed);response=row['response_ids'];width=row.get('canvas_length',256)
    if not response:raise ValueError('empty response')
    block=rng.randrange(math.ceil(len(response)/width));start=block*width
    clean=response[start:start+width];canvas=clean+[0]*(width-len(clean))
    probability=rng.choice([.25,.5,.75,1.])
    positions=[i for i in range(len(clean)) if rng.random()<probability]
    if not positions:positions=[rng.randrange(len(clean))]
    for i in positions:canvas[i]=rng.randrange(vocab_size)
    # Previous completed canvases are clean context. Current/future targets are not.
    return row['input_ids']+response[:start],canvas,positions,[clean[i] for i in positions]

def generation_loss(model,row,seed,chunk_size=16):
    device=model.lm_head.weight.device
    prefix,canvas,positions,targets=noisy_block(row,seed,model.config.text_config.vocab_size)
    with torch.no_grad():
        cache=model.model.encoder(input_ids=torch.tensor([prefix],device=device)).past_key_values
    hidden=model.model.decoder(decoder_input_ids=torch.tensor([canvas],device=device),
                               past_key_values=cache).last_hidden_state[0,positions]
    targets=torch.tensor(targets,device=device);total=hidden.new_zeros((),dtype=torch.float32)
    def token_loss(h,y):
        z=model.lm_head(h).float();cap=model.final_logit_softcapping
        z=cap*torch.tanh(z/cap)
        return torch.nn.functional.cross_entropy(z,y,reduction='sum')
    for begin in range(0,len(positions),chunk_size):
        h=hidden[begin:begin+chunk_size];y=targets[begin:begin+chunk_size]
        total=total+(checkpoint(token_loss,h,y,use_reentrant=False) if torch.is_grad_enabled() else token_loss(h,y))
    return total/len(positions)

def evaluate_generation(model,rows,out,name,count=32):
    model.eval();values=[]
    with torch.no_grad():
        for row in rows[:count]:
            values.append({'id':row['id'],'loss':float(generation_loss(model,row,1907))})
    import json
    result={'metric':'mean corrupted-token full-vocabulary CE; not autoregressive perplexity',
            'records':len(values),'mean_loss':sum(r['loss'] for r in values)/len(values),'rows':values}
    (out/f'{name}-generation.json').write_text(json.dumps(result,indent=2))
    return result
