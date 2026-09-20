"""Training math checks; GPU is not needed, but PyTorch is optional locally."""
import importlib.util
from pathlib import Path
import pytest

torch=pytest.importorskip('torch')
spec=importlib.util.spec_from_file_location('slot_pilot',Path(__file__).parents[1]/'scripts/train_slot_pilot.py')
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)

def test_zero_initialized_adapter_preserves_base_and_learns():
    torch.manual_seed(3)
    base=torch.nn.Linear(8,12,bias=False).requires_grad_(False)
    layer=pilot.LoRA(base,rank=2)
    x=torch.randn(3,8)
    assert torch.equal(layer(x),base(x))
    layer(x).square().mean().backward()
    assert base.weight.grad is None
    assert layer.B.grad.abs().sum()>0
    opt=torch.optim.SGD([layer.A,layer.B],lr=.1);opt.step();opt.zero_grad()
    layer(x).square().mean().backward()
    assert layer.A.grad.abs().sum()>0
    assert not torch.equal(layer(x),base(x))

def test_allowed_label_projection_matches_full_softcapped_logits():
    torch.manual_seed(5)
    hidden=torch.randn(7);head=torch.randn(100,7);allowed=[9,4,33]
    full=30*torch.tanh(torch.nn.functional.linear(hidden,head)/30)
    restricted=30*torch.tanh(torch.nn.functional.linear(hidden,head[allowed])/30)
    torch.testing.assert_close(restricted,full[allowed])
    # Conditioning full-vocabulary probabilities on allowed labels is equivalent.
    p=full.softmax(0)[allowed];p=p/p.sum()
    torch.testing.assert_close(restricted.softmax(0),p)
