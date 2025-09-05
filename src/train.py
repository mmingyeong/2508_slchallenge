# train.py
# -*- coding: utf-8 -*-
"""
Training script for binary strong-lensing classification using ConvNeXt V2.

- data_loader.py must provide `get_dataloaders(class_paths, ...)`
- model.py must provide `convnextv2_atto/nano/tiny` (in_chans=1, num_classes=1)

This version adds preprocessing toggles (padding / normalization) via CLI and
saves a concise results.json for easy comparison across runs.
"""

import os
import csv
import time
import math
import json
import argparse
import logging
import random
import numpy as np
from typing import Dict
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from data_loader import get_dataloaders
from model import convnextv2_atto, convnextv2_nano, convnextv2_tiny


# ---------------------------
# Utilities
# ---------------------------
def setup_logger(log_file: str | None = None,
                 level: int = logging.INFO) -> logging.Logger:
    """
    Configure a logger that:
      - prints to console without breaking tqdm progress bars
      - optionally logs to a rotating file (5MB x 3 backups)
      - quiets noisy third-party loggers
    """
    class TqdmHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                try:
                    from tqdm import tqdm as _tqdm
                    _tqdm.write(msg)
                except Exception:
                    print(msg)
            except Exception:
                self.handleError(record)

    logger = logging.getLogger("train")
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False

    datefmt = "%Y-%m-%d %H:%M:%S"
    console_fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt=datefmt
    )
    file_fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] "
            "[pid:%(process)d] [%(filename)s:%(lineno)d] %(message)s",
        datefmt=datefmt
    )

    ch = TqdmHandler()
    ch.setLevel(level)
    ch.setFormatter(console_fmt)
    logger.addHandler(ch)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        from logging.handlers import RotatingFileHandler
        fh = RotatingFileHandler(
            log_file, mode="w", maxBytes=5 * 1024 * 1024, backupCount=3,
            encoding="utf-8", delay=True
        )
        fh.setLevel(level)
        fh.setFormatter(file_fmt)
        logger.addHandler(fh)

    # quiet noisy libs
    logging.getLogger("data_loader").setLevel(logging.ERROR)
    logging.getLogger("astropy").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logger.info("Logger initialized" + (f" -> {log_file}" if log_file else " (console only)"))
    return logger


class EarlyStopping:
    """Early stopping based on validation loss."""
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = float("inf")
        self.stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_amp_ctx(device: torch.device):
    return torch.amp.autocast(device_type=device.type) if device.type == "cuda" else nullcontext()


# ---------------------------
# Train / Validate
# ---------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, criterion, log_every: int = 100):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="Train", leave=False)
    for i, batch in enumerate(pbar, 1):
        imgs, labels, _ = batch  # (B,1,H,W), (B,)
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with get_amp_ctx(device):
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        probs = torch.sigmoid(logits).detach().cpu().numpy().squeeze()
        preds = (probs > 0.5).astype(np.int32)
        all_preds.append(preds)
        all_labels.append(labels.detach().cpu().numpy().squeeze())

        if i % log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    all_preds = np.concatenate(all_preds) if len(all_preds) else np.array([])
    all_labels = np.concatenate(all_labels) if len(all_labels) else np.array([])

    acc = accuracy_score(all_labels, all_preds) if all_labels.size else 0.0
    avg_loss = running_loss / max(1, len(loader))
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, device, criterion, desc="Val"):
    model.eval()
    running_loss = 0.0
    all_probs, all_preds, all_labels = [], [], []

    pbar = tqdm(loader, desc=desc, leave=False)
    for imgs, labels, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

        with get_amp_ctx(device):
            logits = model(imgs)
            loss = criterion(logits, labels)

        running_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy().squeeze()
        preds = (probs > 0.5).astype(np.int32)
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy().squeeze())

    all_probs = np.concatenate(all_probs) if len(all_probs) else np.array([])
    all_preds = np.concatenate(all_preds) if len(all_preds) else np.array([])
    all_labels = np.concatenate(all_labels) if len(all_labels) else np.array([])

    avg_loss = running_loss / max(1, len(loader))
    acc = accuracy_score(all_labels, all_preds) if all_labels.size else 0.0
    try:
        auc = roc_auc_score(all_labels, all_probs) if all_labels.size else 0.0
    except ValueError:
        auc = float("nan")

    return avg_loss, acc, auc


# ---------------------------
# Main
# ---------------------------
def main(args):
    logger = setup_logger(os.path.join(args.save_dir, "train.log"))
    set_seed(args.seed)

    logger.info("🚀 Configuration")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # Build dataloaders
    class_paths: Dict[str, str] = {
        "slsim_lenses": args.slsim_lenses,
        "slsim_nonlenses": args.slsim_nonlenses,
        "hsc_lenses": args.hsc_lenses,
        "hsc_nonlenses": args.hsc_nonlenses,
    }

    train_loader, val_loader, test_loader = get_dataloaders(
        class_paths=class_paths,
        batch_size=args.batch_size,
        split=(args.train_frac, args.val_frac, args.test_frac),
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=True,
        augment_train=not args.no_augment,
        take_train_fraction=args.take_train_frac,
        take_val_fraction=getattr(args, "take_val_fraction", None),
        take_test_fraction=getattr(args, "take_test_fraction", None),

        # NEW: pass-through toggles/knobs to data_loader
        apply_padding=args.apply_padding,
        out_size_when_padded=args.out_size_when_padded,
        apply_normalization=args.apply_normalization,
        clip_q=args.clip_q if (args.clip_q is None or args.clip_q >= 0) else None,
        low_clip_q=args.low_clip_q,
        use_mad=args.use_mad,
    )

    # Model
    device = torch.device(args.device)
    model_factory = {
        "atto": convnextv2_atto,
        "nano": convnextv2_nano,
        "tiny": convnextv2_tiny,
    }[args.model_size]

    model = model_factory(
        in_chans=1,
        num_classes=1,
        drop_path_rate=args.drop_path,
    ).to(device)

    # Loss / Optim / Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cosine:
        warmup_epochs = max(0, args.warmup_epochs)
        total_epochs  = args.epochs

        def lr_lambda(epoch_idx: int):
            if epoch_idx < warmup_epochs:
                return (epoch_idx + 1) / max(1, warmup_epochs)
            t = (epoch_idx - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * t))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    scaler = torch.amp.GradScaler(device.type if device.type == "cuda" else "cpu")
    early = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    # Paths
    os.makedirs(args.save_dir, exist_ok=True)
    best_ckpt = os.path.join(args.save_dir, "best.pt")
    last_ckpt = os.path.join(args.save_dir, "last.pt")
    csv_log = os.path.join(args.save_dir, "training_log.csv")

    # CSV header
    with open(csv_log, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc",
                         "val_loss", "val_acc", "val_auc", "lr"])

    # ---------------------------
    # Epoch loop
    # ---------------------------
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tic = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device, criterion, args.log_every
        )

        val_loss, val_acc, val_auc = evaluate(model, val_loader, device, criterion, desc="Val")

        lr_now = optimizer.param_groups[0]["lr"]

        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch}, last_ckpt)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_ckpt)
            logger.info(f"✅ Epoch {epoch}: best model updated (val_loss={val_loss:.6f})")

        with open(csv_log, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.4f}",
                             f"{val_loss:.6f}", f"{val_acc:.4f}", f"{val_auc:.6f}", f"{lr_now:.6e}"])

        elapsed = time.time() - tic
        logger.info(
            f"📉 Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.2%} | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.2%} AUC {val_auc:.4f} | "
            f"LR {lr_now:.2e} | {elapsed:.1f}s"
        )

        if scheduler is not None:
            scheduler.step()

        if early.step(val_loss):
            logger.info(f"⏹️ Early stopping at epoch {epoch}")
            break

    # ---------------------------
    # Final evaluation on test set
    # ---------------------------
    logger.info("🔍 Evaluating on test set (best checkpoint)...")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_loss, test_acc, test_auc = evaluate(model, test_loader, device, criterion, desc="Test")
    logger.info(f"✅ Test | Loss {test_loss:.4f} | Acc {test_acc:.2%} | AUC {test_auc:.4f}")

    # Save a compact results file for cross-run comparison
    results = {
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_auc": test_auc,
        "config": {
            "model_size": args.model_size,
            "epochs": args.epochs,
            "seed": args.seed,
            "apply_padding": args.apply_padding,
            "out_size_when_padded": args.out_size_when_padded,
            "apply_normalization": args.apply_normalization,
            "clip_q": args.clip_q if (args.clip_q is None or args.clip_q >= 0) else None,
            "low_clip_q": args.low_clip_q,
            "use_mad": args.use_mad,
            "take_train_frac": args.take_train_frac,
            "take_val_fraction": getattr(args, "take_val_fraction", None),
            "take_test_fraction": getattr(args, "take_test_fraction", None),
        }
    }
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ConvNeXt V2 on FITS (binary classification)")

    # Data paths (required)
    parser.add_argument("--slsim_lenses",      type=str, required=True)
    parser.add_argument("--slsim_nonlenses",   type=str, required=True)
    parser.add_argument("--hsc_lenses",        type=str, required=True)
    parser.add_argument("--hsc_nonlenses",     type=str, required=True)

    # Dataloader options
    parser.add_argument("--batch_size",        type=int, default=128)
    parser.add_argument("--num_workers",       type=int, default=8)
    parser.add_argument("--no_augment",        action="store_true", help="Disable train-time rotation/flip")
    parser.add_argument("--take_train_frac",   type=float, default=None, help="e.g., 0.10 to use 10% of train only")
    parser.add_argument("--take_val_fraction", type=float, default=None)
    parser.add_argument("--take_test_fraction",type=float, default=None)

    # Split
    parser.add_argument("--train_frac",        type=float, default=0.70)
    parser.add_argument("--val_frac",          type=float, default=0.15)
    parser.add_argument("--test_frac",         type=float, default=0.15)

    # Model
    parser.add_argument("--model_size",        type=str, default="atto", choices=["atto", "nano", "tiny"])
    parser.add_argument("--drop_path",         type=float, default=0.1)

    # Optim
    parser.add_argument("--lr",                type=float, default=3e-4)
    parser.add_argument("--weight_decay",      type=float, default=5e-2)
    parser.add_argument("--cosine",            action="store_true", help="Use cosine LR with warmup")
    parser.add_argument("--warmup_epochs",     type=int, default=2)

    # Train
    parser.add_argument("--epochs",            type=int, default=20)
    parser.add_argument("--patience",          type=int, default=10)
    parser.add_argument("--min_delta",         type=float, default=0.0)
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--device",            type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_every",         type=int, default=200, help="Steps between loss prints in training")

    # Save
    parser.add_argument("--save_dir",          type=str, default="./checkpoints_convnextv2")

    # === Preprocessing toggles (pass-through to data_loader) ===
    parser.add_argument("--apply_padding", action="store_true",
                        help="Center reflect-pad 41x41 -> out_size_when_padded (e.g., 64).")
    parser.add_argument("--out_size_when_padded", type=int, default=64,
                        help="Output size when padding is applied.")
    parser.add_argument("--apply_normalization", action="store_true",
                        help="Background subtraction -> (optional clip) -> z-score (or MAD).")
    parser.add_argument("--clip_q", type=float, default=0.997,
                        help="High-quantile clipping; set <0 to disable (plain z-score).")
    parser.add_argument("--low_clip_q", type=float, default=None,
                        help="Optional low-tail clipping quantile (e.g., 0.005).")
    parser.add_argument("--use_mad", action="store_true",
                        help="Use robust median/MAD instead of mean/std for z-score.")

    args = parser.parse_args()

    # sanitize clip_q: allow negative to mean None
    if args.clip_q is not None and args.clip_q < 0:
        args.clip_q = None

    main(args)
