import pytest
import torch
from model import Kronos
from supervised.model import SupervisedKronos, from_specification, tiny_config


@pytest.mark.parametrize('task',['point','sequence'])
def test_hybrid_padding_reconstruction_and_gradient(task):
    torch.manual_seed(71)
    model=SupervisedKronos(Kronos(**tiny_config()),3,task=task,head_type='hybrid',sequence_pooling='attention').eval()
    x=torch.randn(2,12,3);lengths=torch.tensor([7,12])
    original=model(x,lengths)
    padded=x.clone();padded[0,7:]=1000
    changed=model(padded,lengths)
    torch.testing.assert_close(original[0,:7] if task=='point' else original[0],changed[0,:7] if task=='point' else changed[0])
    restored=from_specification(model.specification()).eval();restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(x,lengths),original)
    original.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.raw_classifier.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.backbone.transformer.parameters())


def test_hybrid_point_never_reads_future():
    torch.manual_seed(73)
    model=SupervisedKronos(Kronos(**tiny_config()),3,head_type='hybrid').eval()
    x=torch.randn(2,17,3);lengths=torch.tensor([17,17])
    before=model(x,lengths);x[:,9:]+=100
    torch.testing.assert_close(before[:,:9],model(x,lengths)[:,:9])


def test_hybrid_sequence_constant_inputs_have_finite_gradients():
    model=SupervisedKronos(Kronos(**tiny_config()),2,task='sequence',head_type='hybrid')
    x=torch.ones(2,9,2,requires_grad=True)
    model(x,torch.tensor([5,9])).sum().backward()
    assert torch.isfinite(x.grad).all()
