# evaluate.py
# -*- coding: utf-8 -*-
"""
Evaluation utilities for binary strong-lensing classification.

Inputs
------
- From NPY: labels.npy, probs.npy, (optional) preds.npy
- From CSV: pred_*.csv with columns: path, domain, label, prob, pred

Outputs
-------
- metrics_{tag}.json : global metrics (and optionally per-domain)
- metrics_{tag}.csv  : same metrics in tabular form
- (optional) plots:
    * roc_{tag}.png
    * pr_{tag}.png
    * calibration_{tag}.png

Metrics
-------
- Accuracy, Precision, Recall, F1 (binary & macro)
- Balanced Accuracy, Specificity (TNR)
- ROC-AUC, PR-AUC (Average Precision)
- Matthews Corr. Coef. (MCC)
- Brier score
- Expected Calibration Error (ECE, reliability)
- Optimal threshold (Youden’s J) [optional]

Usage
-----
python evaluate.py \
  --tag test \
  --from_npy --labels labels_test.npy --probs probs_test.npy --preds preds_test.npy \
  --out_dir ./eval_outputs

python evaluate.py \
  --tag test \
  --from_csv --csv pred_test.csv \
  --per_domain \
  --optimize_threshold \
  --plot \
  --out_dir ./eval_outputs
"""

import os
import json
import csv
import math
import argparse
import logging
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef, brier_score_loss
)

# ---------------------------
# Logging
# ---------------------------
def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("evaluate")


# ---------------------------
# I/O helpers
# ---------------------------
def load_from_csv(csv_path: str):
    """Load label/prob/pred (+ path/domain) from a CSV produced by predict.py."""
    labels, probs, preds, paths, domains = [], [], [], [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row["label"]))
            probs.append(float(row["prob"]))
            # pred may be absent (e.g., if you only saved probs)
            if "pred" in row and row["pred"] not in (None, "", "nan"):
                preds.append(int(row["pred"]))
            else:
                preds.append(None)
            paths.append(row.get("path", ""))
            domains.append(row.get("domain", ""))
    # If preds had Nones, convert later
    return np.array(labels), np.array(probs), np.array(preds, dtype=object), paths, domains


def load_from_npy(labels_path: str, probs_path: str, preds_path: Optional[str] = None):
    """Load arrays from npy files."""
    labels = np.load(labels_path)
    probs  = np.load(probs_path)
    preds  = np.load(preds_path) if preds_path else None
    return labels, probs, preds


# ---------------------------
# Metrics
# ---------------------------
def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> Tuple[float, dict]:
    """
    Compute ECE with equal-width bins in probability space.

    Returns
    -------
    ece : float
    details : dict with per-bin stats (bin_edges, conf, acc, count)
    """
    eps = 1e-12
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins) - 1
    ece = 0.0
    details = {"bins": [], "conf": [], "acc": [], "count": []}

    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            details["bins"].append((float(bins[b]), float(bins[b+1])))
            details["conf"].append(float("nan"))
            details["acc"].append(float("nan"))
            details["count"].append(0)
            continue
        conf = float(probs[mask].mean())
        acc  = float((labels[mask] == (probs[mask] >= 0.5)).mean())
        gap = abs(acc - conf)
        ece += (count / max(1, len(labels))) * gap

        details["bins"].append((float(bins[b]), float(bins[b+1])))
        details["conf"].append(conf)
        details["acc"].append(acc)
        details["count"].append(count)

    return float(ece), details


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True Negative Rate."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return tn / max(1, (tn + fp))


def compute_metrics(labels: np.ndarray, probs: np.ndarray, preds: Optional[np.ndarray] = None,
                    threshold: float = 0.5) -> Dict:
    """
    Compute a suite of binary classification metrics.
    If preds is None, it will be derived from probs via the given threshold.
    """
    if preds is None or (preds.dtype == object):
        preds = (probs >= threshold).astype(int)

    # Basic
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)

    # Macro (useful if later you move to multi-class)
    prec_m = precision_score(labels, preds, average="macro", zero_division=0)
    rec_m  = recall_score(labels, preds, average="macro", zero_division=0)
    f1_m   = f1_score(labels, preds, average="macro", zero_division=0)

    # Other diagnostics
    bal_acc = (rec + specificity_score(labels, preds)) / 2.0
    spec    = specificity_score(labels, preds)
    mcc     = matthews_corrcoef(labels, preds)
    brier   = brier_score_loss(labels, probs)

    # Curves
    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(labels, probs)
    except ValueError:
        pr_auc = float("nan")

    # Calibration
    ece, ece_detail = expected_calibration_error(labels, probs, n_bins=15)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "precision_macro": prec_m,
        "recall_macro": rec_m,
        "f1_macro": f1_m,
        "balanced_accuracy": bal_acc,
        "specificity": spec,
        "mcc": mcc,
        "brier": brier,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "ece": ece,
        "threshold": threshold,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "ece_detail": ece_detail
    }


def optimize_threshold(labels: np.ndarray, probs: np.ndarray) -> Tuple[float, Dict]:
    """
    Find optimal threshold by maximizing Youden's J = TPR - FPR.
    Returns the best threshold and metrics at that threshold.
    """
    # Candidate thresholds from unique probabilities
    ths = np.unique(probs)
    if len(ths) > 4096:  # subsample for speed if extremely dense
        ths = np.linspace(0, 1, 4096)

    best_th, best_j, best_metrics = 0.5, -1.0, {}
    for th in ths:
        preds = (probs >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
        tpr = tp / max(1, (tp + fn))
        fpr = fp / max(1, (fp + tn))
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_th = float(th)

    best_metrics = compute_metrics(labels, probs, None, threshold=best_th)
    return best_th, best_metrics


# ---------------------------
# Plotting
# ---------------------------
def plot_roc(labels, probs, out_path):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out_path); plt.close()


def plot_pr(labels, probs, out_path):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    plt.figure()
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve"); plt.legend(loc="lower left")
    plt.tight_layout(); plt.savefig(out_path); plt.close()


def plot_calibration(labels, probs, out_path, n_bins: int = 15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins) - 1
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        xs.append(probs[mask].mean())
        ys.append((labels[mask] == (probs[mask] >= 0.5)).mean())
        ns.append(mask.sum())

    plt.figure()
    plt.plot([0,1], [0,1], "--", label="Perfect")
    plt.scatter(xs, ys, s=np.array(ns) * 1.5, alpha=0.7, label="Bins")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.tight_layout(); plt.savefig(out_path); plt.close()


# ---------------------------
# Main
# ---------------------------
def main(args):
    logger = setup_logger(os.path.join(args.out_dir, f"evaluate_{args.tag}.log"))
    os.makedirs(args.out_dir, exist_ok=True)

    # Load
    if args.from_csv:
        labels, probs, preds, paths, domains = load_from_csv(args.csv)
        logger.info(f"Loaded CSV: N={labels.size} from {args.csv}")
        if (preds.dtype == object) or (preds.size == 0) or (preds[0] is None):
            preds = None
    elif args.from_npy:
        labels, probs, preds = load_from_npy(args.labels, args.probs, args.preds)
        paths, domains = [], []
        logger.info(f"Loaded NPY: N={labels.size} (labels/probs) | preds={'None' if preds is None else preds.shape}")
    else:
        raise ValueError("Specify either --from_csv or --from_npy.")

    # Sanity
    if labels.shape[0] != probs.shape[0]:
        raise ValueError("labels and probs must have same length.")
    if preds is not None and preds.shape[0] != labels.shape[0]:
        logger.warning("preds length differs from labels; preds will be ignored.")
        preds = None

    # Threshold (optional optimization)
    if args.optimize_threshold:
        best_th, best_metrics = optimize_threshold(labels, probs)
        th = best_th
        logger.info(f"Optimal threshold (Youden J): {th:.4f}")
        global_metrics = best_metrics
    else:
        th = args.threshold
        global_metrics = compute_metrics(labels, probs, preds, threshold=th)

    # Per-domain metrics
    metrics = {"global": global_metrics}
    if args.per_domain and len(domains) == labels.size and len(domains) > 0:
        doms = np.array(domains)
        for dom in sorted(np.unique(doms)):
            mask = doms == dom
            if mask.sum() == 0:
                continue
            m = compute_metrics(labels[mask], probs[mask], None, threshold=th)
            metrics[f"domain::{dom}"] = m

    # Save metrics json/csv
    json_path = os.path.join(args.out_dir, f"metrics_{args.tag}.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics JSON → {json_path}")

    # Flatten to CSV
    csv_path = os.path.join(args.out_dir, f"metrics_{args.tag}.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["scope", "metric", "value"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scope, mdict in metrics.items():
            for k, v in mdict.items():
                if k == "ece_detail":
                    continue
                writer.writerow({"scope": scope, "metric": k, "value": v})
    logger.info(f"Saved metrics CSV → {csv_path}")

    # Plots
    if args.plot:
        plot_roc(labels, probs, os.path.join(args.out_dir, f"roc_{args.tag}.png"))
        plot_pr(labels, probs, os.path.join(args.out_dir, f"pr_{args.tag}.png"))
        plot_calibration(labels, probs, os.path.join(args.out_dir, f"calibration_{args.tag}.png"))
        logger.info("Saved ROC/PR/Calibration plots.")

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate binary classifier predictions.")
    parser.add_argument("--tag", type=str, default="test", help="A tag used in output filenames.")

    # Input modality
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--from_csv", action="store_true", help="Load predictions from CSV file.")
    g.add_argument("--from_npy", action="store_true", help="Load predictions from NPY files.")

    # CSV inputs
    parser.add_argument("--csv", type=str, help="Path to CSV with columns: path,domain,label,prob,pred")

    # NPY inputs
    parser.add_argument("--labels", type=str, help="Path to labels.npy")
    parser.add_argument("--probs", type=str, help="Path to probs.npy")
    parser.add_argument("--preds", type=str, default=None, help="Path to preds.npy (optional)")

    # Options
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold when preds not provided.")
    parser.add_argument("--optimize_threshold", action="store_true", help="Use Youden’s J to pick optimal threshold.")
    parser.add_argument("--per_domain", action="store_true", help="Report metrics per domain if available.")
    parser.add_argument("--plot", action="store_true", help="Save ROC/PR/Calibration plots.")
    parser.add_argument("--out_dir", type=str, default="./eval_outputs")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    np.random.seed(args.seed)
    main(args)
