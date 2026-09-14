from __future__ import annotations

import argparse
import csv
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .attacks import adaptive_pgd, apply_upp, fgsm, frequency_attack, geometry_aware, optimize_upp, physics_proxy
from .checkpoints import build_pipeline, load_checkpoint, save_checkpoint
from .evaluation import evaluate_attack
from .geometry import DifferentiableRadon, fbp_reconstruct, make_angles
from .loaders import make_dataset, make_loader
from .metrics import confidence_and_entropy, input_saliency, saliency_instability
from .training import fit
from .utils import get_device, save_json, seed_everything


def _data_kwargs(args):
    return dict(data_root=args.data_root, manifest=args.manifest, image_size=args.image_size, input_mode=args.input_mode)


def _add_data_args(p):
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-root", type=str, help="Folder with train/valid/test/{benign,malignant}")
    src.add_argument("--manifest", type=str, help="CSV with path,label,patient_id,split")
    p.add_argument("--input-mode", choices=["normalized", "hu"], default="normalized")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)


def _train(args):
    seed_everything(args.seed)
    device = get_device(args.device)
    train_ds = make_dataset(split="train", **_data_kwargs(args))
    val_ds = make_dataset(split="validation", **_data_kwargs(args))
    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    angles = make_angles(args.views, device=device)
    radon = DifferentiableRadon(angles).to(device)
    pipeline = build_pipeline(args.arch, args.rsdf).to(device)
    history = fit(pipeline, train_loader, val_loader, radon, angles, device, epochs=args.epochs, lr=args.lr)
    save_checkpoint(args.output, pipeline, arch=args.arch, use_rsdf=args.rsdf, n_views=args.views, extra={"history": history})
    save_json({"checkpoint": str(args.output), "history": history}, Path(args.output).with_suffix(".history.json"))
    print(json.dumps(history[-1], indent=2))


def _load_eval_context(args, split="test"):
    seed_everything(args.seed)
    device = get_device(args.device)
    pipeline, ckpt = load_checkpoint(args.checkpoint, device)
    views = args.views if getattr(args, "views", None) else int(ckpt.get("n_views", 180))
    angles = make_angles(views, device=device)
    radon = DifferentiableRadon(angles).to(device)
    ds = make_dataset(split=split, **_data_kwargs(args))
    loader = make_loader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return device, pipeline, angles, radon, loader, ckpt


def _write_rows(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    else:
        keys = sorted({k for row in rows for k in row})
        with output.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"Saved {output}")


def _attack_family(args):
    device, pipeline, angles, radon, loader, _ = _load_eval_context(args, split=args.split)
    rows = []
    attacks = {
        "fgsm": (fgsm, {"eps": args.eps}),
        "physics_proxy": (physics_proxy, {"eps": args.eps, "n0": args.n0, "z_alpha": args.z_alpha}),
        "geometry_aware": (geometry_aware, {"eps": args.eps}),
        "frequency_domain": (frequency_attack, {"eps": args.eps, "band": args.frequency_band}),
    }
    for name, (fn, kw) in attacks.items():
        metrics = evaluate_attack(pipeline, loader, radon, angles, fn, device, compute_saliency=not args.skip_saliency, **kw)
        rows.append({"attack": name, **metrics})

    # UPP is optimized on train/development data and then frozen.
    opt_ds = make_dataset(split=args.upp_opt_split, **_data_kwargs(args))
    opt_loader = make_loader(opt_ds, batch_size=16, shuffle=True, num_workers=args.num_workers)
    upp = optimize_upp(opt_loader, radon, angles, pipeline, args.eps, epochs=args.upp_epochs, step_size=args.upp_step, device=device)
    def upp_attack(g, labels, angles, pipeline, eps):
        return apply_upp(g, upp)
    metrics = evaluate_attack(pipeline, loader, radon, angles, upp_attack, device, eps=args.eps, compute_saliency=not args.skip_saliency)
    rows.append({"attack": "upp", **metrics})
    _write_rows(rows, args.output)


def _frequency(args):
    device, pipeline, angles, radon, loader, _ = _load_eval_context(args, split=args.split)
    rows=[]
    metrics = evaluate_attack(pipeline, loader, radon, angles, fgsm, device, eps=args.eps)
    rows.append({"variant":"unrestricted_fgsm", **metrics})
    base_drop = metrics["accuracy_drop"]
    for band in ("low", "high"):
        m = evaluate_attack(pipeline, loader, radon, angles, frequency_attack, device, eps=args.eps, band=band)
        m["relative_effectiveness"] = m["accuracy_drop"] / base_drop if base_drop else float("nan")
        rows.append({"variant":band, **m})
    _write_rows(rows,args.output)


def _budget(args):
    device, pipeline, angles, radon, loader, _ = _load_eval_context(args, split=args.split)
    rows=[]
    for eps in args.eps_values:
        for name, fn in (("fgsm",fgsm),("physics_proxy",physics_proxy)):
            m=evaluate_attack(pipeline,loader,radon,angles,fn,device,eps=eps)
            rows.append({"epsilon":eps,"attack":name,**m})
    _write_rows(rows,args.output)


def _adaptive(args):
    device, pipeline, angles, radon, loader, _ = _load_eval_context(args, split=args.split)
    rows=[]
    m=evaluate_attack(pipeline,loader,radon,angles,fgsm,device,eps=args.eps)
    rows.append({"attack":"fgsm","steps":1,"restarts":1,**m})
    for steps in args.steps:
        m=evaluate_attack(pipeline,loader,radon,angles,adaptive_pgd,device,eps=args.eps,steps=steps,restarts=args.restarts,alpha=0.1*args.eps)
        rows.append({"attack":f"pgd-{steps}","steps":steps,"restarts":args.restarts,**m})
    _write_rows(rows,args.output)


def _confidence(args):
    device, pipeline, angles, radon, loader, _ = _load_eval_context(args, split=args.split)
    rows=[]
    zero_conf=None
    for eps in args.eps_values:
        if eps == 0:
            # clean-vs-clean baseline via FGSM eps=0
            m=evaluate_attack(pipeline,loader,radon,angles,fgsm,device,eps=0.0)
        else:
            m=evaluate_attack(pipeline,loader,radon,angles,fgsm,device,eps=eps)
        if zero_conf is None:
            zero_conf=m["adversarial_confidence"]
        m["confidence_retained_percent"] = 100.0*m["adversarial_confidence"]/zero_conf if zero_conf else float("nan")
        rows.append({"epsilon":eps,**m})
    _write_rows(rows,args.output)


def _upp_transfer(args):
    device, pipeline, _, _, eval_loader, _ = _load_eval_context(args, split=args.split)
    opt_angles = make_angles(args.optimize_views, device=device)
    opt_radon = DifferentiableRadon(opt_angles).to(device)
    opt_ds = make_dataset(split=args.upp_opt_split, **_data_kwargs(args))
    opt_loader = make_loader(opt_ds,batch_size=16,shuffle=True,num_workers=args.num_workers)
    upp=optimize_upp(opt_loader,opt_radon,opt_angles,pipeline,args.eps,epochs=args.upp_epochs,step_size=args.upp_step,device=device)
    torch.save(upp.cpu(), Path(args.output).with_suffix(".upp.pt"))
    rows=[]
    for views in args.view_counts:
        angles=make_angles(views,device=device); radon=DifferentiableRadon(angles).to(device)
        def attack(g,labels,angles,pipeline,eps): return apply_upp(g,upp)
        m=evaluate_attack(pipeline,eval_loader,radon,angles,attack,device,eps=args.eps)
        rows.append({"views":views,**m})
    _write_rows(rows,args.output)


def _sparse_view(args):
    seed_everything(args.seed)
    device=get_device(args.device)
    train_ds=make_dataset(split="train",**_data_kwargs(args)); val_ds=make_dataset(split="validation",**_data_kwargs(args))
    eval_ds=make_dataset(split=args.split,**_data_kwargs(args))
    train_loader=make_loader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
    val_loader=make_loader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
    eval_loader=make_loader(eval_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
    rows=[]
    ckpt_dir=Path(args.checkpoint_dir); ckpt_dir.mkdir(parents=True,exist_ok=True)
    for views in args.view_counts:
        angles=make_angles(views,device=device); radon=DifferentiableRadon(angles).to(device)
        pipeline=build_pipeline(args.arch,args.rsdf).to(device)
        history=fit(pipeline,train_loader,val_loader,radon,angles,device,epochs=args.epochs,lr=args.lr)
        save_checkpoint(ckpt_dir/f"{args.arch}_{'rsdf' if args.rsdf else 'baseline'}_{views}views.pt",pipeline,arch=args.arch,use_rsdf=args.rsdf,n_views=views,extra={"history":history})
        m=evaluate_attack(pipeline,eval_loader,radon,angles,fgsm,device,eps=args.eps)
        rows.append({"views":views,**m})
    _write_rows(rows,args.output)



def _architecture_ablation(args):
    seed_everything(args.seed)
    device = get_device(args.device)
    eval_ds = make_dataset(split=args.split, **_data_kwargs(args))
    eval_loader = make_loader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    opt_ds = make_dataset(split=args.upp_opt_split, **_data_kwargs(args))
    opt_loader = make_loader(opt_ds, batch_size=16, shuffle=True, num_workers=args.num_workers)
    rows=[]
    for label, checkpoint in [
        ("cnn3", args.cnn3_checkpoint),
        ("cnn3_rsdf", args.rsdf_checkpoint),
        ("cnn5", args.cnn5_checkpoint),
    ]:
        pipeline, ckpt = load_checkpoint(checkpoint, device)
        views = int(ckpt.get("n_views", 180))
        angles = make_angles(views, device=device)
        radon = DifferentiableRadon(angles).to(device)
        for attack_name, fn, kw in [
            ("fgsm", fgsm, {"eps":args.eps}),
            ("physics_proxy", physics_proxy, {"eps":args.eps}),
            ("geometry_aware", geometry_aware, {"eps":args.eps}),
            ("frequency_domain", frequency_attack, {"eps":args.eps,"band":args.frequency_band}),
        ]:
            m=evaluate_attack(pipeline,eval_loader,radon,angles,fn,device,**kw)
            rows.append({"architecture":label,"attack":attack_name,**m})
        upp=optimize_upp(opt_loader,radon,angles,pipeline,args.eps,epochs=args.upp_epochs,step_size=args.upp_step,device=device)
        def upp_attack(g,labels,angles,pipeline,eps): return apply_upp(g,upp)
        m=evaluate_attack(pipeline,eval_loader,radon,angles,upp_attack,device,eps=args.eps)
        rows.append({"architecture":label,"attack":"upp",**m})
    _write_rows(rows,args.output)


def build_parser():
    p=argparse.ArgumentParser(description="Projection-domain CT adversarial robustness experiments")
    sp=p.add_subparsers(dest="command",required=True)

    t=sp.add_parser("train",help="Train a paper-style clean-reconstruction classifier")
    _add_data_args(t); t.add_argument("--views",type=int,default=180); t.add_argument("--arch",choices=["cnn3","cnn5"],default="cnn3"); t.add_argument("--rsdf",action="store_true")
    t.add_argument("--epochs",type=int,default=20); t.add_argument("--lr",type=float,default=1e-3); t.add_argument("--output",required=True); t.set_defaults(func=_train)

    def eval_base(name,helptext):
        q=sp.add_parser(name,help=helptext); _add_data_args(q); q.add_argument("--checkpoint",required=True); q.add_argument("--views",type=int); q.add_argument("--split",default="test"); q.add_argument("--output",required=True); return q

    q=eval_base("attack-family","Evaluate the five primary attack families")
    q.add_argument("--eps",type=float,default=0.25); q.add_argument("--n0",type=float,default=1e5); q.add_argument("--z-alpha",type=float,default=1.96); q.add_argument("--frequency-band",choices=["low","mid","high","adaptive"],default="adaptive")
    q.add_argument("--upp-opt-split",default="train"); q.add_argument("--upp-epochs",type=int,default=20); q.add_argument("--upp-step",type=float,default=0.01); q.add_argument("--skip-saliency",action="store_true"); q.set_defaults(func=_attack_family)

    q=eval_base("frequency-ablation","Unrestricted FGSM vs low/high detector-frequency attacks"); q.add_argument("--eps",type=float,default=0.25); q.set_defaults(func=_frequency)
    q=eval_base("budget-sweep","FGSM and physics proxy across epsilon budgets"); q.add_argument("--eps-values",type=float,nargs="+",default=[0.05,0.10,0.15,0.20,0.30]); q.set_defaults(func=_budget)
    q=eval_base("adaptive-pgd","Clean-correct ASR for FGSM and adaptive PGD"); q.add_argument("--eps",type=float,default=0.25); q.add_argument("--steps",type=int,nargs="+",default=[10,20,40]); q.add_argument("--restarts",type=int,default=3); q.set_defaults(func=_adaptive)
    q=eval_base("confidence-sweep","FGSM confidence retention, entropy, and flip-rate sweep"); q.add_argument("--eps-values",type=float,nargs="+",default=[0.0,0.2,0.4,0.6,0.65,0.75]); q.set_defaults(func=_confidence)
    q=eval_base("upp-transfer","Optimize a UPP at one view count and transfer across geometries"); q.add_argument("--eps",type=float,default=0.25); q.add_argument("--optimize-views",type=int,default=180); q.add_argument("--view-counts",type=int,nargs="+",default=[180,120,90,60,30]); q.add_argument("--upp-opt-split",default="train"); q.add_argument("--upp-epochs",type=int,default=20); q.add_argument("--upp-step",type=float,default=0.01); q.set_defaults(func=_upp_transfer)

    q=sp.add_parser("architecture-ablation",help="Compare CNN3, CNN3+RSDF and CNN5 across the five primary attacks")
    _add_data_args(q); q.add_argument("--cnn3-checkpoint",required=True); q.add_argument("--rsdf-checkpoint",required=True); q.add_argument("--cnn5-checkpoint",required=True); q.add_argument("--split",default="test"); q.add_argument("--eps",type=float,default=0.25); q.add_argument("--frequency-band",choices=["low","mid","high","adaptive"],default="adaptive"); q.add_argument("--upp-opt-split",default="train"); q.add_argument("--upp-epochs",type=int,default=20); q.add_argument("--upp-step",type=float,default=0.01); q.add_argument("--output",required=True); q.set_defaults(func=_architecture_ablation)

    q=sp.add_parser("sparse-view",help="Train/evaluate a separate configuration at each view count, as in supplied scripts")
    _add_data_args(q); q.add_argument("--view-counts",type=int,nargs="+",default=[30,60,90,120,180]); q.add_argument("--eps",type=float,default=0.25); q.add_argument("--epochs",type=int,default=20); q.add_argument("--lr",type=float,default=1e-3); q.add_argument("--arch",choices=["cnn3","cnn5"],default="cnn3"); q.add_argument("--rsdf",action="store_true"); q.add_argument("--split",default="test"); q.add_argument("--checkpoint-dir",default="outputs/checkpoints/sparse_view"); q.add_argument("--output",required=True); q.set_defaults(func=_sparse_view)
    return p


def main(argv=None):
    args=build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
