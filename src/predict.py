# predict.py
# -*- coding: utf-8 -*-
"""
Predict / evaluate script for binary strong-lensing classification using ConvNeXt V2.

Now supports passing preprocessing toggles to data_loader:
- padding: --apply_padding, --out_size_when_padded
- normalization: --apply_normalization, --clip_q, --low_clip_q, --use_mad
- smoothing: --smoothing_mode {none,gaussian,guided}, --gaussian_sigma, --guided_radius, --guided_eps

NEW:
- Accepts either directory inputs (--slsim_lenses, ...) or manifest list files
  (--slsim_lenses_list, ...) for large-scale inference without creating symlinks.
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
    amp_ctx = torch.amp.autocast(device_type=device.type) if device.type == "cuda" else nullcontext()

    seen = 0
    for imgs, labels, metas in pbar:
        if max_samples is not None and seen >= max_samples:
            break

        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

        with amp_ctx:
            logits = model(imgs)

        probs = torch.sigmoid(logits).cpu().numpy().squeeze()
        preds = (probs > 0.5).astype(np.int32)
        ys    = labels.cpu().numpy().squeeze()

        probs = np.atleast_1d(probs)
        preds = np.atleast_1d(preds)
        ys    = np.atleast_1d(ys)

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

        for m in metas:
            all_paths.append(m["path"])
            all_domains.append(m["domain"])

        seen += probs.shape[0]

    labels = np.concatenate(all_labels) if all_labels else np.array([])
    probs  = np.concatenate(all_probs)  if all_probs  else np.array([])
    preds  = np.concatenate(all_preds)  if all_preds  else np.array([])

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
# Utilities for dir-or-list inputs
# ---------------------------
def _read_list(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        items = [ln.strip() for ln in f if ln.strip()]
    return items or None


def _validate_dir_or_list(logger: logging.Logger, class_dirs: Dict[str, Optional[str]],
                          class_lists: Dict[str, Optional[List[str]]]) -> None:
    """Ensure each class has at least one source (dir or list)."""
    for k in ("slsim_lenses", "slsim_nonlenses", "hsc_lenses", "hsc_nonlenses"):
        if not (class_dirs.get(k) or class_lists.get(k)):
            raise ValueError(f"Provide either --{k} (directory) or --{k}_list (manifest).")
    # Informative logging
    for k in class_dirs:
        src = f"LIST({len(class_lists[k])})" if class_lists.get(k) else f"DIR({class_dirs.get(k)})"
        logger.info(f"Input source for {k}: {src}")


# ---------------------------
# Main
# ---------------------------
def main(args):
    logger = setup_logger(os.path.join(args.output_dir, "predict.log"))
    device = torch.device(args.device)

    # Gather dir-or-list sources
    class_dirs: Dict[str, Optional[str]] = {
        "slsim_lenses":    args.slsim_lenses,
        "slsim_nonlenses": args.slsim_nonlenses,
        "hsc_lenses":      args.hsc_lenses,
        "hsc_nonlenses":   args.hsc_nonlenses,
    }
    class_lists: Dict[str, Optional[List[str]]] = {
        "slsim_lenses":    _read_list(args.slsim_lenses_list),
        "slsim_nonlenses": _read_list(args.slsim_nonlenses_list),
        "hsc_lenses":      _read_list(args.hsc_lenses_list),
        "hsc_nonlenses":   _read_list(args.hsc_nonlenses_list),
    }
    _validate_dir_or_list(logger, class_dirs, class_lists)
    logger.info("📦 Building dataloaders (dir-or-list)…")

    # ----- SAFE READ of optional smoothing args -----
    smoothing_mode_raw = getattr(args, "smoothing_mode", "none")
    smoothing_mode = None if (smoothing_mode_raw is None or str(smoothing_mode_raw).lower() == "none") \
                     else str(smoothing_mode_raw).lower()

    gaussian_sigma = float(getattr(args, "gaussian_sigma", 1.0))
    guided_radius  = int(getattr(args, "guided_radius", 2))
    guided_eps     = float(getattr(args, "guided_eps", 1e-3))

    # clip_q 음수 → None
    clip_q = getattr(args, "clip_q", 0.997)
    if clip_q is not None and clip_q < 0:
        clip_q = None

    # IMPORTANT: requires data_loader.get_dataloaders signature updated accordingly.
    train_loader, val_loader, test_loader = get_dataloaders(
        class_dirs=class_dirs,                 # NEW
        class_lists=class_lists,               # NEW
        batch_size=args.batch_size,
        split=(args.train_frac, args.val_frac, args.test_frac),
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=True,
        augment_train=False,  # prediction → no augment
        # debug sampling
        take_train_fraction=getattr(args, "take_train_fraction", None),
        take_val_fraction=getattr(args, "take_val_fraction", None),
        take_test_fraction=getattr(args, "take_test_fraction", None),
        # --- preprocessing toggles (pass-through) ---
        apply_padding=getattr(args, "apply_padding", False),
        out_size_when_padded=getattr(args, "out_size_when_padded", 64),
        apply_normalization=getattr(args, "apply_normalization", False),
        clip_q=clip_q,
        low_clip_q=getattr(args, "low_clip_q", None),
        use_mad=getattr(args, "use_mad", False),
        # smoothing
        smoothing_mode=smoothing_mode,            # None | "gaussian" | "guided"
        gaussian_sigma=gaussian_sigma,
        guided_radius=guided_radius,
        guided_eps=guided_eps,
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

    # --- Data dirs (optional; dir-or-list model) ---
    parser.add_argument("--slsim_lenses",      type=str, required=False,
                        help="Directory of SLSim lenses (if not using *_list).")
    parser.add_argument("--slsim_nonlenses",   type=str, required=False,
                        help="Directory of SLSim non-lenses (if not using *_list).")
    parser.add_argument("--hsc_lenses",        type=str, required=False,
                        help="Directory of HSC lenses (if not using *_list).")
    parser.add_argument("--hsc_nonlenses",     type=str, required=False,
                        help="Directory of HSC non-lenses (if not using *_list).")

    # --- NEW: manifest (one path per line) ---
    parser.add_argument("--slsim_lenses_list",    type=str, default=None,
                        help="Text file: one FITS path per line for SLSim lenses.")
    parser.add_argument("--slsim_nonlenses_list", type=str, default=None,
                        help="Text file: one FITS path per line for SLSim non-lenses.")
    parser.add_argument("--hsc_lenses_list",      type=str, default=None,
                        help="Text file: one FITS path per line for HSC lenses.")
    parser.add_argument("--hsc_nonlenses_list",   type=str, default=None,
                        help="Text file: one FITS path per line for HSC non-lenses.")

    # Which split
    parser.add_argument("--which",             type=str, default="test",
                        help="train | val | test | all")

    # Loader (split)
    parser.add_argument("--batch_size",        type=int, default=128)
    parser.add_argument("--num_workers",       type=int, default=8)
    parser.add_argument("--train_frac",        type=float, default=0.70)
    parser.add_argument("--val_frac",          type=float, default=0.15)
    parser.add_argument("--test_frac",         type=float, default=0.15)
    parser.add_argument("--seed",              type=int, default=42)

    # Debug subsampling
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
    parser.add_argument("--max_samples",       type=int, default=None)

    # ===== Preprocessing toggles (pass-through to data_loader) =====
    # Padding
    parser.add_argument("--apply_padding", action="store_true",
                        help="Center pad 41x41 -> out_size_when_padded (constant background).")
    parser.add_argument("--out_size_when_padded", type=int, default=64)

    # Normalization
    parser.add_argument("--apply_normalization", action="store_true",
                        help="Background subtraction -> (optional clip) -> z-score (or MAD).")
    parser.add_argument("--clip_q", type=float, default=0.997,
                        help="High-quantile clipping; set <0 to disable (plain z-score).")
    parser.add_argument("--low_clip_q", type=float, default=None,
                        help="Optional low-tail clipping quantile (e.g., 0.005).")
    parser.add_argument("--use_mad", action="store_true",
                        help="Use robust median/MAD instead of mean/std for z-score.")

    # Smoothing
    parser.add_argument("--smoothing_mode", type=str, default="none",
                        choices=["none", "gaussian", "guided"],
                        help="Apply smoothing before padding: none | gaussian | guided.")
    parser.add_argument("--gaussian_sigma", type=float, default=1.0,
                        help="Gaussian sigma in pixels (only if smoothing_mode=gaussian).")
    parser.add_argument("--guided_radius",  type=int,   default=2,
                        help="Guided filter radius in pixels (only if smoothing_mode=guided).")
    parser.add_argument("--guided_eps",     type=float, default=1e-3,
                        help="Guided filter regularization epsilon (only if smoothing_mode=guided).")

    args = parser.parse_args()

    if args.clip_q is not None and args.clip_q < 0:
        args.clip_q = None

    main(args)
