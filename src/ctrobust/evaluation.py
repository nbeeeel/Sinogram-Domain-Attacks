from __future__ import annotations

import numpy as np
import torch

from .geometry import fbp_reconstruct
from .metrics import confidence_and_entropy, input_saliency, saliency_instability


def evaluate_attack(
    pipeline,
    loader,
    radon,
    angles,
    attack_fn,
    device,
    *,
    compute_saliency: bool = False,
    **attack_kwargs,
):
    """Evaluate clean/adversarial accuracy, flip rate, ASR, confidence, entropy and optional Eq. (28)."""
    pipeline.eval()
    n = clean_correct = adv_correct = flips = clean_correct_attacked_wrong = 0
    clean_conf, adv_conf, clean_ent, adv_ent = [], [], [], []
    saliency_values = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            g = radon(images)
            clean_recon = fbp_reconstruct(g, angles)
            clean_logits = pipeline(clean_recon)
            clean_pred = clean_logits.argmax(1)
            cconf, cent = confidence_and_entropy(clean_logits)

        adv_g = attack_fn(g, labels, angles, pipeline, **attack_kwargs)
        with torch.no_grad():
            adv_recon = fbp_reconstruct(adv_g, angles)
            adv_logits = pipeline(adv_recon)
            adv_pred = adv_logits.argmax(1)
            aconf, aent = confidence_and_entropy(adv_logits)

        if compute_saliency:
            s_clean = input_saliency(pipeline, clean_recon, labels)
            s_adv = input_saliency(pipeline, adv_recon, labels)
            saliency_values.extend(saliency_instability(s_clean, s_adv).tolist())

        clean_ok = clean_pred == labels
        n += labels.numel()
        clean_correct += clean_ok.sum().item()
        adv_correct += (adv_pred == labels).sum().item()
        flips += (clean_pred != adv_pred).sum().item()
        clean_correct_attacked_wrong += (clean_ok & (adv_pred != labels)).sum().item()
        clean_conf.append(cconf.cpu()); adv_conf.append(aconf.cpu())
        clean_ent.append(cent.cpu()); adv_ent.append(aent.cpu())

    if n == 0:
        raise ValueError("Evaluation loader is empty")
    cc = torch.cat(clean_conf); ac = torch.cat(adv_conf)
    ce = torch.cat(clean_ent); ae = torch.cat(adv_ent)
    clean_acc = clean_correct / n
    adv_acc = adv_correct / n
    result = {
        "clean_accuracy": clean_acc,
        "adversarial_accuracy": adv_acc,
        "accuracy_drop": clean_acc - adv_acc,
        "flip_rate": flips / n,
        "asr_clean_correct": clean_correct_attacked_wrong / clean_correct if clean_correct else float("nan"),
        "clean_confidence": cc.mean().item(),
        "adversarial_confidence": ac.mean().item(),
        "confidence_change": (cc - ac).mean().item(),
        "clean_entropy": ce.mean().item(),
        "adversarial_entropy": ae.mean().item(),
    }
    if compute_saliency:
        result["saliency_instability"] = float(np.mean(saliency_values)) if saliency_values else float("nan")
    return result
