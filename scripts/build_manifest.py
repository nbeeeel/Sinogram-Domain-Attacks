#!/usr/bin/env python3
"""Build a paper-style patient-wise split manifest from slice metadata.

Input CSV columns:
  path, patient_id, median_malignancy

Rules from the paper:
  median <= 2 -> benign (0)
  median >= 4 -> malignant (1)
  median == 3 -> excluded

The exact nodule extraction/consensus construction is not described in enough detail
in the manuscript; this script starts from already-selected nodule-centred slices.
"""
from __future__ import annotations

import argparse, csv, random
from pathlib import Path


def main():
    p=argparse.ArgumentParser(); p.add_argument("input_csv"); p.add_argument("output_csv"); p.add_argument("--seed",type=int,default=0)
    args=p.parse_args()
    rows=[]
    with open(args.input_csv,newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            score=float(r["median_malignancy"])
            if score <= 2: label=0
            elif score >= 4: label=1
            else: continue
            rows.append({**r,"label":label})
    patients=sorted({r["patient_id"] for r in rows}); random.Random(args.seed).shuffle(patients)
    n=len(patients); n_train=round(0.70*n); n_valid=round(0.15*n)
    split={pid:("train" if i<n_train else "valid" if i<n_train+n_valid else "test") for i,pid in enumerate(patients)}
    out=Path(args.output_csv); out.parent.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0].keys())+["split"] if rows else ["path","patient_id","median_malignancy","label","split"]
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({**r,"split":split[r["patient_id"]]})
    print(f"Wrote {len(rows)} labeled slices from {n} patients to {out}")

if __name__ == "__main__": main()
