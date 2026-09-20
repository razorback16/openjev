import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

torch=pytest.importorskip('torch')

def module(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).parents[1]/'scripts'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

replay=module('generation_replay')

def test_noise_has_no_future_context_or_padding_targets():
    row={'input_ids':[40,41],'response_ids':list(range(1,12)),'canvas_length':4}
    starts=set()
    for seed in range(40):
        prefix,canvas,positions,targets=replay.noisy_block(row,seed,vocab_size=50)
        start=len(prefix)-2;starts.add(start)
        assert prefix==row['input_ids']+row['response_ids'][:start]
        assert targets==[row['response_ids'][start+i] for i in positions]
        assert all(i<min(4,11-start) for i in positions)
        assert len(canvas)==4 and positions
    assert starts=={0,4,8}

def test_replay_schedule_has_exact_mix():
    assert sum(replay.is_replay(i,.2) for i in range(100))==20
    with pytest.raises(ValueError):replay.is_replay(0,1.1)

def test_generation_loss_backpropagates_and_chunking_agrees():
    torch.manual_seed(7)
    class Encoder:
        def __call__(self,**kwargs):return SimpleNamespace(past_key_values=None)
    class Decoder(torch.nn.Module):
        def __init__(self):super().__init__();self.embedding=torch.nn.Embedding(50,8)
        def forward(self,decoder_input_ids,**kwargs):return SimpleNamespace(last_hidden_state=self.embedding(decoder_input_ids))
    decoder=Decoder();head=torch.nn.Linear(8,50,bias=False).requires_grad_(False)
    model=SimpleNamespace(model=SimpleNamespace(encoder=Encoder(),decoder=decoder),lm_head=head,
                          final_logit_softcapping=30,config=SimpleNamespace(text_config=SimpleNamespace(vocab_size=50)))
    row={'input_ids':[1,2],'response_ids':[3,4,5,6],'canvas_length':4}
    a=replay.generation_loss(model,row,3,chunk_size=1);a.backward()
    grad=decoder.embedding.weight.grad.clone();assert grad.abs().sum()>0
    decoder.zero_grad();b=replay.generation_loss(model,row,3,chunk_size=16);b.backward()
    torch.testing.assert_close(a,b);torch.testing.assert_close(grad,decoder.embedding.weight.grad)
    assert head.weight.grad is None

def test_thinking_prefix_matches_production_and_has_no_duplicate_scaffold():
    from transformers import AutoTokenizer
    from openjev.engine import Engine
    from openjev.config import Settings
    tok=AutoTokenizer.from_pretrained(Settings().tokenizer,local_files_only=True)
    engine=Engine(Settings(),tok)
    row={'id':'example','group_id':'group','state':'The account balance is positive.',
         'questions':{'q1':{'type':'noul','instructions':'Is the balance positive?'}},
         'targets':{'q1':'yes'},'provenance':{'family':'test'}}
    schema=engine.build_schema(row['questions'])
    prompt=engine.chat_prompt_ids(engine.system_text(schema['questions'],schema['format']),row['state'],thinking=True)
    prefix=prompt+engine.thought_open+engine.enc('The state says the balance is positive.')+engine.thought_close
    compiled=module('collect_thinking_replay').compile_thinking(engine,row,prefix)
    expected,slots=engine.resolve_template(schema['questions'],schema['format'],head=[])
    assert compiled['input_ids']==prefix and compiled['template']==expected and compiled['slots']==slots
    assert compiled['targets']==[0]
    assert compiled['thought_replay']['input_ids']==prompt
    assert compiled['thought_replay']['response_ids']==prefix[len(prompt):]
    asyncio.run(engine.close())
