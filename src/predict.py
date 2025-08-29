# predict.py
# -*- coding: utf-8 -*-
"""
Predict / evaluate script for binary strong-lensing classification using ConvNeXt V2.
"""

import os
import csv
import argparse
import logging
from typing import Dict, Tuple, List, Optional
from contextlib import nullcontext

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from data_loader import get_dataloaders
from model import convnextv2_atto, convnextv2_nano, convnextv2_tiny


# ---------------------------
# Logging
# ---------------------------
def setup_logger(log_file: str | None = None) -> logging.Logger:
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [predict] %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("predict")


# ---------------------------
# Eval helper
# ---------------------------
@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    desc: str = "Predict",
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Returns:
      labels (N,), probs (N,), preds (N,), paths (list[str]), domains (list[str])
    """
    model.eval()
    all_labels, all_probs, all_preds = [], [], []
    all_paths, all_domains = [], []

    pbar = tqdm(loader, desc=desc, leave=False)
    amp_ctx = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()

    seen = 0
    for imgs, labels, metas in pbar:
        # limit early if max_samples is small and we already have enough
        if max_samples is not None and seen >= max_samples:
            break

        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

        with amp_ctx:
            logits = model(imgs)

        probs = torch.sigmoid(logits).cpu().numpy().squeeze()
        preds = (probs > 0.5).astype(np.int32)
        ys    = labels.cpu().numpy().squeeze()

        # batch-size가 1일 수도 있으니 일괄적으로 np.atleast_1d 적용
        probs = np.atleast_1d(probs)
        preds = np.atleast_1d(preds)
        ys    = np.atleast_1d(ys)

        # 남은 quota만큼만 취득
        if max_samples is not None:
            remain = max_samples - seen
            if probs.shape[0] > remain:
                probs = probs[:remain]
                preds = preds[:remain]
                ys    = ys[:remain]
                metas = metas[:remain]

        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(ys)

        # safe_collate → metas is list[dict]
        for m in metas:
            all_paths.append(m["path"])
            all_domains.append(m["domain"])

        seen += probs.shape[0]

    labels = np.concatenate(all_labels) if all_labels else np.array([])
    probs  = np.concatenate(all_probs)  if all_probs  else np.array([])
    preds  = np.concatenate(all_preds)  if all_preds  else np.array([])

    # 혹시라도 길이 불일치 방지
    n = min(len(labels), len(probs), len(preds), len(all_paths), len(all_domains))
    return labels[:n], probs[:n], preds[:n], all_paths[:n], all_domains[:n]


def build_model(model_size: str, drop_path: float, device: torch.device) -> torch.nn.Module:
    factory = {
        "atto": convnextv2_atto,
        "nano": convnextv2_nano,
        "tiny": convnextv2_tiny,
    }
    model = factory[model_size](in_chans=1, num_classes=1, drop_path_rate=drop_path).to(device)
    return model


# ---------------------------
# Main
# ---------------------------
def main(args):
    logger = setup_logger(os.path.join(args.output_dir, "predict.log"))
    device = torch.device(args.device)

    # Build dataloaders (same split as train)
    class_paths: Dict[str, str] = {
        "slsim_lenses": args.slsim_lenses,
        "slsim_nonlenses": args.slsim_nonlenses,
        "hsc_lenses": args.hsc_lenses,
        "hsc_nonlenses": args.hsc_nonlenses,
    }
    logger.info("📦 Building dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        class_paths=class_paths,
        batch_size=args.batch_size,
        split=(args.train_frac, args.val_frac, args.test_frac),
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=True,
        augment_train=False,  # prediction → no augment
        # 디버그용 샘플링 전달
        take_train_fraction=getattr(args, "take_train_fraction", None),
        take_val_fraction=getattr(args, "take_val_fraction", None),
        take_test_fraction=getattr(args, "take_test_fraction", None),
    )
    logger.info(f"Split sizes -> train:{len(train_loader.dataset)}  "
                f"val:{len(val_loader.dataset)}  test:{len(test_loader.dataset)}")

    # Choose which split(s)
    which = args.which.lower()
    selected = []
    if which in ("train", "all"):
        selected.append(("train", train_loader))
    if which in ("val", "valid", "validation", "all"):
        selected.append(("val", val_loader))
    if which in ("test", "all"):
        selected.append(("test", test_loader))
    if not selected:
        raise ValueError(f"Invalid --which '{args.which}'. Choose from train/val/test/all.")

    # Model + checkpoint
    logger.info("🧠 Loading model...")
    model = build_model(args.model_size, args.drop_path, device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)

    os.makedirs(args.output_dir, exist_ok=True)

    # Inference per split
    for tag, loader in selected:
        logger.info(f"🚀 Inference on split: {tag}")
        labels, probs, preds, paths, domains = run_inference(
            model, loader, device, desc=f"Predict-{tag}", max_samples=getattr(args, "max_samples", None)
        )

        # Metrics
        acc = accuracy_score(labels, preds) if labels.size else float("nan")
        try:
            auc = roc_auc_score(labels, probs) if labels.size else float("nan")
        except ValueError:
            auc = float("nan")
        logger.info(f"✅ {tag.upper()} | Acc={acc:.4f} | AUC={auc:.4f} | N={labels.size}")

        # Save CSV
        csv_path = os.path.join(args.output_dir, f"pred_{tag}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "domain", "label", "prob", "pred"])
            for p, d, y, pr, pd in zip(paths, domains, labels.tolist(), probs.tolist(), preds.tolist()):
                writer.writerow([p, d, y, pr, pd])
        logger.info(f"💾 Saved per-sample CSV → {csv_path}")

        # Save NPY
        np.save(os.path.join(args.output_dir, f"labels_{tag}.npy"), labels)
        np.save(os.path.join(args.output_dir, f"probs_{tag}.npy"), probs)
        np.save(os.path.join(args.output_dir, f"preds_{tag}.npy"), preds)
        logger.info(f"💾 Saved NPY arrays for {tag} split")

    logger.info("🎯 Done.")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict/evaluate ConvNeXt V2 on FITS (binary classification)")
    # Data dirs
    parser.add_argument("--slsim_lenses",      type=str, required=True)
    parser.add_argument("--slsim_nonlenses",   type=str, required=True)
    parser.add_argument("--hsc_lenses",        type=str, required=True)
    parser.add_argument("--hsc_nonlenses",     type=str, required=True)
    # Which split
    parser.add_argument("--which",             type=str, default="test",
                        help="train | val | test | all")
    # Loader
    parser.add_argument("--batch_size",        type=int, default=128)
    parser.add_argument("--num_workers",       type=int, default=8)
    parser.add_argument("--train_frac",        type=float, default=0.70)
    parser.add_argument("--val_frac",          type=float, default=0.15)
    parser.add_argument("--test_frac",         type=float, default=0.15)
    parser.add_argument("--seed",              type=int, default=42)
    # 디버그 서브샘플링
    parser.add_argument("--take_train_fraction", type=float, default=None)
    parser.add_argument("--take_val_fraction",   type=float, default=None)
    parser.add_argument("--take_test_fraction",  type=float, default=None)
    # Model
    parser.add_argument("--model_path",        type=str, required=True)
    parser.add_argument("--model_size",        type=str, default="atto", choices=["atto", "nano", "tiny"])
    parser.add_argument("--drop_path",         type=float, default=0.0)
    # Runtime
    parser.add_argument("--device",            type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir",        type=str, default="./pred_outputs")
    # 최대 N개만 추론 (split별)
    parser.add_argument("--max_samples",       type=int, default=None)
    args = parser.parse_args()
    main(args)
