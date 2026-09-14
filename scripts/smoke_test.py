#!/usr/bin/env python3
import torch
from ctrobust.geometry import DifferentiableRadon, fbp_reconstruct, make_angles
from ctrobust.models import DiagnosticPipeline
from ctrobust.attacks import fgsm

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x=torch.rand(2,1,32,32,device=device); y=torch.tensor([0,1],device=device)
    angles=make_angles(12,device=device); radon=DifferentiableRadon(angles).to(device); model=DiagnosticPipeline().to(device)
    g=radon(x); r=fbp_reconstruct(g,angles); adv=fgsm(g,y,angles,model,0.05)
    print("image",tuple(x.shape),"sinogram",tuple(g.shape),"recon",tuple(r.shape),"max_delta",float((adv-g).abs().max()))

if __name__ == "__main__": main()
