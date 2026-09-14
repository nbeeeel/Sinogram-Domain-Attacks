import math
import numpy as np
import torch

from ctrobust.attacks import adaptive_pgd, fgsm, frequency_attack, frequency_mask, geometry_aware, physics_proxy
from ctrobust.data import preprocess_hu
from ctrobust.geometry import DifferentiableRadon, fbp_reconstruct, make_angles
from ctrobust.metrics import saliency_instability
from ctrobust.models import CTClassifier3, CTClassifier5, DiagnosticPipeline, RSDF


def _context(size=16, views=8):
    torch.manual_seed(0)
    x=torch.rand(2,1,size,size)
    y=torch.tensor([0,1])
    angles=make_angles(views)
    radon=DifferentiableRadon(angles)
    g=radon(x)
    model=DiagnosticPipeline()
    return x,y,angles,radon,g,model


def test_angles_are_half_open():
    a=make_angles(180)
    assert a[0].item() == 0.0
    assert a[-1].item() < math.pi
    assert torch.allclose(a[1]-a[0], torch.tensor(math.pi/180), atol=1e-7)


def test_geometry_shapes_and_gradients():
    x,y,a,r,g,m=_context()
    assert g.shape == (2,1,16,8)
    g = g.detach().requires_grad_(True)
    rec=fbp_reconstruct(g,a)
    assert rec.shape == x.shape
    loss=m(rec).sum()
    grad=torch.autograd.grad(loss,g)[0]
    assert torch.isfinite(grad).all()


def test_reported_parameter_counts():
    assert sum(p.numel() for p in CTClassifier3().parameters()) == 23426
    assert sum(p.numel() for p in RSDF().parameters()) == 305
    assert sum(p.numel() for p in CTClassifier5().parameters()) == 392834
    assert sum(p.numel() for p in DiagnosticPipeline(use_rsdf=True).parameters()) == 23731


def test_attack_linf_bounds():
    x,y,a,r,g,m=_context()
    eps=0.05
    for fn,kwargs in [
        (fgsm,{}),
        (physics_proxy,{}),
        (geometry_aware,{}),
        (frequency_attack,{"band":"high"}),
    ]:
        adv=fn(g,y,a,m,eps,**kwargs)
        assert float((adv-g).abs().max()) <= eps + 1e-5


def test_pgd_linf_bound():
    x,y,a,r,g,m=_context(size=12,views=6)
    eps=0.05
    adv=adaptive_pgd(g,y,a,m,eps,steps=2,restarts=2)
    assert float((adv-g).abs().max()) <= eps + 1e-6


def test_frequency_bands_do_not_overlap():
    low=frequency_mask(64,"low").bool()
    mid=frequency_mask(64,"mid").bool()
    high=frequency_mask(64,"high").bool()
    assert not torch.any(low & mid)
    assert not torch.any(mid & high)
    assert not torch.any(low & high)
    assert torch.all(low | mid | high)


def test_hu_preprocessing_endpoints():
    x=np.array([[-1200,400],[0,800]],dtype=np.float32)
    y=preprocess_hu(x,image_size=2)
    assert y.min() >= 0 and y.max() <= 1
    assert np.isclose(y[0,0],0.0)
    assert np.isclose(y[0,1],1.0)


def test_identical_saliency_has_zero_instability():
    s=torch.rand(2,1,16,16)
    v=saliency_instability(s,s)
    assert np.allclose(v,0,atol=1e-6)
