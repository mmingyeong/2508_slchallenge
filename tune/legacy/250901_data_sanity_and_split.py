#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PBS-friendly script: Data sanity check & split freeze for Strong Lens Challenge.

What this does (simple & robust):
1) Collect all FITS paths via src/data_loader.collect_files().
2) (Optional) Scan a subset for corrupted FITS using _read_fits_image_41x41().
3) Build fixed Train/Val/Test splits with get_dataloaders(
       split=(0.70,0.15,0.15), seed=42
   ).
4) Log counts and class/domain distributions per split.
5) Save outputs:
   - <out_dir>/data_check.log
   - <out_dir>/summary.json
   - <out_dir>/bad_files.txt  (if any)

Usage (example)
---------------
python data_sanity_and_split.py \
  --slsim_lenses      /caefs/data/IllustrisTNG/slchallenge/slsim_lenses/slsim_lenses \
  --slsim_nonlenses   /caefs/data/IllustrisTNG/slchallenge/slsim_nonlenses/slsim_nonlenses \
  --hsc_lenses        /caefs/data/IllustrisTNG/slchallenge/hsc_lenses/hsc_lenses \
  --hsc_nonlenses     /caefs/data/IllustrisTNG/slchallenge/hsc_nonlenses/hsc_nonlenses \
  --scan_fraction 0.02 \
  --num_workers 8 \
  --out_dir ./_data_check

Note
----
- This file is assumed to live at: 2508_slchallence/tune/data_sanity_and_split.py
- The code automatically adds 2508_slchallence/src to sys.path.
"""

import os
import sys
import json
import time
import argparse
import logging
import socket
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch


# ---------------------------
# Logging
# ---------------------------
def setup_logger(log_path: str) -> logging.Logger:
    """Set up console+file logger (INFO)."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("data_check")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def main():
    # Resolve project root from this file location:
    # .../2508_slchallence/tune/data_sanity_and_split.py  -> project_root = .../2508_slchallence
    this_file = Path(__file__).resolve()
    project_root_default = this_file.parents[1]  # 2508_slchallence
    src_dir = project_root_default / "src"

    parser = argparse.ArgumentParser(description="Data sanity & split-freeze (PBS-friendly, simple)")
    # Data dirs (required)
    parser.add_argument("--slsim_lenses",      type=str, required=True)
    parser.add_argument("--slsim_nonlenses",   type=str, required=True)
    parser.add_argument("--hsc_lenses",        type=str, required=True)
    parser.add_argument("--hsc_nonlenses",     type=str, required=True)

    # Split & workers
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac",   type=float, default=0.15)
    parser.add_argument("--test_frac",  type=float, default=0.15)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--num_workers", type=int,  default=max(1, (os.cpu_count() or 8)//2))

    # Corruption scan subset
    parser.add_argument("--scan_fraction", type=float, default=1.0,
                        help="Fraction of files to scan for corruption (0<f<=1).")
    parser.add_argument("--scan_limit", type=int, default=0,
                        help="Absolute cap on files to scan (0=no cap).")

    # Output
    parser.add_argument("--out_dir", type=str, default=str(project_root_default / "tune" / "_data_check"))

    # Optional override (normally not needed since we auto-detect)
    parser.add_argument("--project_root", type=str, default=str(project_root_default))

    args = parser.parse_args()

    # Paths
    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(out_dir / "data_check.log")

    logger = setup_logger(log_path)

    # Minimal environment banner
    logger.info("=== Environment ===")
    logger.info(f"Host       : {socket.gethostname()}")
    logger.info(f"Python     : {sys.version.split()[0]}")
    logger.info(f"Torch      : {torch.__version__}")
    logger.info(f"CUDA avail : {torch.cuda.is_available()}")
    logger.info(f"Project    : {project_root}")
    logger.info(f"Src dir    : {project_root / 'src'}")
    logger.info(f"Workers    : {args.num_workers}")

    # Import from src/
    sys.path.insert(0, str(project_root / "src"))
    try:
        from data_loader import collect_files, _read_fits_image_41x41, get_dataloaders
    except Exception as e:
        logger.error(f"Failed to import project modules from src/: {e}")
        sys.exit(1)

    # Build class paths
    class_paths = {
        "slsim_lenses":    args.slsim_lenses,
        "slsim_nonlenses": args.slsim_nonlenses,
        "hsc_lenses":      args.hsc_lenses,
        "hsc_nonlenses":   args.hsc_nonlenses,
    }

    t0 = time.time()

    # 1) Collect files
    files, labels, domains = collect_files(class_paths)
    total = len(files)
    lab_counts = dict(Counter(labels))
    dom_counts = dict(Counter(domains))
    logger.info("=== Collected files ===")
    logger.info(f"TOTAL files  : {total}")
    logger.info(f"Labels count : {lab_counts}")
    logger.info(f"Domains count: {dom_counts}")

    # 2) Optional corruption scan (subset)
    scan_files = files
    if 0 < args.scan_fraction < 1.0:
        rng = np.random.default_rng(args.seed)
        k = max(1, int(total * args.scan_fraction))
        idx = rng.choice(total, size=k, replace=False)
        scan_files = [files[i] for i in idx]
    if args.scan_limit and args.scan_limit > 0:
        scan_files = scan_files[:args.scan_limit]

    bad_files = []
    if len(scan_files) > 0:
        logger.info(f"=== Scanning subset for corruption ===")
        logger.info(f"Scan subset : {len(scan_files)} (of {total})")

        def _check(fp: str):
            ok = _read_fits_image_41x41(fp) is not None
            return fp, ok

        done = 0
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [ex.submit(_check, fp) for fp in scan_files]
            for fut in as_completed(futures):
                fp, ok = fut.result()
                if not ok:
                    bad_files.append(fp)
                done += 1
                if done % 5000 == 0:
                    logger.info(f" scanned {done}/{len(scan_files)} ...")

        (out_dir / "bad_files.txt").write_text("\n".join(bad_files))
        logger.info(f"Bad files found: {len(bad_files)} → {out_dir / 'bad_files.txt'}")
    else:
        logger.info("Skipping corruption scan (no files selected).")

    # 3) Build fixed splits (seeded)
    logger.info("=== Building splits via get_dataloaders ===")
    train_loader, val_loader, test_loader = get_dataloaders(
        class_paths=class_paths,
        batch_size=128,  # not important here
        split=(args.train_frac, args.val_frac, args.test_frac),
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=True,
        augment_train=False,
        take_train_fraction=None,
        take_val_fraction=None,
        take_test_fraction=None,
    )
    ds_tr, ds_va, ds_te = train_loader.dataset, val_loader.dataset, test_loader.dataset

    def summarize(ds, name: str) -> dict:
        lab = dict(Counter(ds.labels))
        dom = dict(Counter(ds.domains))
        info = {"N": len(ds), "labels": lab, "domains": dom}
        logger.info(f"[{name}] N={info['N']} | labels={info['labels']} | domains={info['domains']}")
        return info

    info_tr = summarize(ds_tr, "train")
    info_va = summarize(ds_va, "val")
    info_te = summarize(ds_te, "test")

    # 4) Save summary
    summary = {
        "env": {
            "host": socket.gethostname(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "num_workers": args.num_workers,
        },
        "inputs": {
            "class_paths": class_paths,
            "split": [args.train_frac, args.val_frac, args.test_frac],
            "seed": args.seed,
            "scan_fraction": args.scan_fraction,
            "scan_limit": args.scan_limit,
        },
        "collection": {
            "total_files": total,
            "labels": lab_counts,
            "domains": dom_counts,
        },
        "scan": {
            "scanned": len(scan_files),
            "bad_files_count": len(bad_files),
            "bad_files_list_path": str(out_dir / "bad_files.txt"),
        },
        "splits": {
            "train": info_tr,
            "val": info_va,
            "test": info_te,
            "sum_N": info_tr["N"] + info_va["N"] + info_te["N"],
        },
        "notes": "Use the same seed & split in all future experiments for comparability.",
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary → {out_dir / 'summary.json'}")
    logger.info(f"Done in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
