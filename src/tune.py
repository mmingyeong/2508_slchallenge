# tune.py (micro-HPO: weight_decay + scheduler family)
# -*- coding: utf-8 -*-
"""
Micro HPO with Optuna aligned to train.py (scheduler ∈ {cosine, step, exp, plateau})

Fixed knobs (override via CLI):
  - arch=atto, batch_size=128, lr=3.5e-4

Search space:
  - weight_decay: log-uniform [wd_min, wd_max]
  - scheduler: categorical in {cosine, step, exp, plateau}
    * cosine : warmup_pct ∈ [warmup_pct_min, warmup_pct_max]; min_lr fixed by CLI
    * step   : step_size ∈ [step_min, step_max] (int), gamma ∈ [step_gamma_min, step_gamma_max]
    * exp    : gamma ∈ [exp_gamma_min, exp_gamma_max]
    * plateau: factor ∈ [plateau_factor_min, plateau_factor_max] (log), patience ∈ [pmin,pmax] (int),
               cooldown ∈ [cmin,cmax] (int), monitor fixed by CLI (loss|auc), min_lr by CLI

Objective:
  - Maximize best validation AUC parsed from training_log.csv

Notes:
  - Keeps your preprocessing/data flags.
  - Designed for single or multi-worker (shared Optuna storage).
"""

import os
import csv
import math
import time
import argparse
import types
from typing import Optional

import numpy as np
import optuna


# ---------------------------
# Helpers
# ---------------------------

def _read_best_val_auc(csv_path: str) -> float:
    """Return max(val_auc) from training_log.csv. If missing/NaN, return 0.5."""
    if not os.path.exists(csv_path):
        return 0.5
    best = float("-inf")
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                v = float(row.get("val_auc", "nan"))
                if math.isfinite(v) and v > best:
                    best = v
            except Exception:
                continue
    return best if best != float("-inf") else 0.5


def _mk_args_for_train(
    *,
    # data
    slsim_lenses: str,
    slsim_nonlenses: str,
    hsc_lenses: str,
    hsc_nonlenses: str,
    # trial-specific
    save_dir: str,
    arch: str,
    lr: float,
    batch_size: int,
    weight_decay: float,
    # schedulers (match train.py)
    scheduler: str,
    warmup_epochs: int,
    min_lr: float,
    step_size: int,
    gamma: float,
    plateau_factor: float,
    plateau_patience: int,
    plateau_cooldown: int,
    plateau_monitor: str,
    # fixed knobs
    drop_path: float,
    epochs: int,
    patience: int,
    seed: int,
    device: str,
    num_workers: int,
    # sampling
    take_train_frac: Optional[float],
    take_val_fraction: Optional[float],
    take_test_fraction: Optional[float],
    # preprocessing
    apply_padding: bool,
    out_size_when_padded: int,
    apply_normalization: bool,
    clip_q: Optional[float],
    low_clip_q: Optional[float],
    use_mad: bool,
    smoothing_mode: str,
    gaussian_sigma: float,
    guided_radius: int,
    guided_eps: float,
) -> types.SimpleNamespace:
    """
    Build a namespace that matches train.py's argparse interface.
    """
    ns = types.SimpleNamespace(
        # data dirs
        slsim_lenses=slsim_lenses,
        slsim_nonlenses=slsim_nonlenses,
        hsc_lenses=hsc_lenses,
        hsc_nonlenses=hsc_nonlenses,
        # dataloader
        batch_size=batch_size,
        num_workers=num_workers,
        no_augment=True,  # HPO 동안 공정 비교를 위해 증강 OFF
        take_train_frac=take_train_frac,
        take_val_fraction=take_val_fraction,
        take_test_fraction=take_test_fraction,
        # split (compat)
        train_frac=0.70,
        val_frac=0.15,
        test_frac=0.15,
        # model/optim
        model_size=arch,
        drop_path=drop_path,
        lr=lr,
        weight_decay=weight_decay,
        # scheduler (exactly as train.py expects)
        scheduler=scheduler,                  # "cosine" | "step" | "exp" | "plateau"
        warmup_epochs=warmup_epochs,          # cosine
        min_lr=min_lr,                        # cosine/plateau LR floor
        step_size=step_size,                  # step
        gamma=gamma,                          # step/exp
        plateau_factor=plateau_factor,        # plateau
        plateau_patience=plateau_patience,    # plateau
        plateau_cooldown=plateau_cooldown,    # plateau
        plateau_monitor=plateau_monitor,      # plateau ("loss"|"auc")
        # train runtime
        epochs=epochs,
        patience=patience,
        min_delta=0.0,
        seed=seed,
        device=device,
        log_every=200,
        # save
        save_dir=save_dir,
        # preprocessing toggles
        apply_padding=apply_padding,
        out_size_when_padded=out_size_when_padded,
        apply_normalization=apply_normalization,
        clip_q=clip_q,
        low_clip_q=low_clip_q,
        use_mad=use_mad,
        # smoothing
        smoothing_mode=smoothing_mode,
        gaussian_sigma=gaussian_sigma,
        guided_radius=guided_radius,
        guided_eps=guided_eps,
    )
    return ns


def _safe_dirname(s: str) -> str:
    s = str(s)
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


# ---------------------------
# Main Optuna driver
# ---------------------------

def main():
    p = argparse.ArgumentParser(description="Optuna micro-HPO (weight_decay + scheduler family) for ConvNeXtV2")

    # Paths to your source tree (so we can import train.py)
    p.add_argument("--src_dir", type=str, default=".", help="Directory containing train.py, etc.")

    # Data dirs (required)
    p.add_argument("--slsim_lenses",      type=str, required=True)
    p.add_argument("--slsim_nonlenses",   type=str, required=True)
    p.add_argument("--hsc_lenses",        type=str, required=True)
    p.add_argument("--hsc_nonlenses",     type=str, required=True)

    # Study / storage
    p.add_argument("--study_name", type=str, default=None, help="Optuna study name (shared across workers)")
    p.add_argument("--storage",    type=str, default=None, help="Optuna storage URL (e.g., postgresql://...)")

    # Trials
    p.add_argument("--n_trials",          type=int, default=None)
    p.add_argument("--trials_per_worker", type=int, default=None)
    p.add_argument("--out_root", type=str, default="./optuna_runs", help="Root directory for trial artifacts")

    # -------- Micro-HPO: fixed knobs (override if needed) --------
    p.add_argument("--fixed_arch",  type=str, default="atto", help="Fixed architecture")
    p.add_argument("--fixed_batch", type=int, default=128,    help="Fixed batch size")
    p.add_argument("--fixed_lr",    type=float, default=3.5e-4, help="Fixed learning rate")

    # -------- Search space (weight decay + scheduler family) --------
    p.add_argument("--wd_min", type=float, default=3e-5)
    p.add_argument("--wd_max", type=float, default=3e-4)
    p.add_argument("--scheduler_choices", type=str, nargs="+",
                   default=["cosine", "step", "exp", "plateau"],
                   help="Schedulers to consider: cosine, step, exp, plateau")

    # Cosine warmup settings
    p.add_argument("--warmup_pct_min", type=float, default=0.03)
    p.add_argument("--warmup_pct_max", type=float, default=0.10)
    p.add_argument("--min_lr",         type=float, default=1e-6)

    # StepLR search bounds
    p.add_argument("--step_min",        type=int,   default=8,   help="min step_size (epochs)")
    p.add_argument("--step_max",        type=int,   default=30,  help="max step_size (epochs)")
    p.add_argument("--step_gamma_min",  type=float, default=0.3, help="min gamma for StepLR")
    p.add_argument("--step_gamma_max",  type=float, default=0.8, help="max gamma for StepLR")

    # ExponentialLR search bounds
    p.add_argument("--exp_gamma_min",   type=float, default=0.90)
    p.add_argument("--exp_gamma_max",   type=float, default=0.999)

    # ReduceLROnPlateau search bounds (monitor is fixed via CLI below)
    p.add_argument("--plateau_factor_min",   type=float, default=0.2)
    p.add_argument("--plateau_factor_max",   type=float, default=0.7)
    p.add_argument("--plateau_patience_min", type=int,   default=2)
    p.add_argument("--plateau_patience_max", type=int,   default=6)
    p.add_argument("--plateau_cooldown_min", type=int,   default=0)
    p.add_argument("--plateau_cooldown_max", type=int,   default=4)
    p.add_argument("--plateau_monitor",      type=str,   default="loss", choices=["loss", "auc"])

    # Fixed training knobs (slightly longer than 20 for stability)
    p.add_argument("--epochs",   type=int,   default=60)
    p.add_argument("--patience", type=int,   default=15)
    p.add_argument("--drop_path", type=float, default=0.05)

    # Sampling / speed knobs
    p.add_argument("--take_train_frac",   type=float, default=0.10)
    p.add_argument("--take_val_fraction", type=float, default=0.20)
    p.add_argument("--take_test_fraction",type=float, default=None)

    # Runtime
    p.add_argument("--seed",    type=int,  default=42)
    p.add_argument("--device",  type=str,  default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu")
    p.add_argument("--workers", type=int,  default=8, help="DataLoader workers")

    # Preprocessing (kept)
    p.add_argument("--apply_padding", action="store_true")
    p.add_argument("--out_size_when_padded", type=int, default=64)
    p.add_argument("--apply_normalization", action="store_true", default=True)
    p.add_argument("--clip_q", type=float, default=0.997)   # set <0 to disable
    p.add_argument("--low_clip_q", type=float, default=None)
    p.add_argument("--use_mad", action="store_true", default=False)
    p.add_argument("--smoothing_mode", type=str, default="gaussian", choices=["none", "gaussian", "guided"])
    p.add_argument("--gaussian_sigma", type=float, default=1.0)
    p.add_argument("--guided_radius",  type=int,   default=2)
    p.add_argument("--guided_eps",     type=float, default=1e-2)

    args = p.parse_args()

    # Prepare imports from src_dir
    import sys
    sys.path.insert(0, os.path.abspath(args.src_dir))
    import train as train_mod  # your existing training entrypoint

    os.makedirs(args.out_root, exist_ok=True)

    # Decide trials count for this worker
    if args.trials_per_worker is not None:
        trials_this_worker = int(args.trials_per_worker)
    elif args.n_trials is not None:
        trials_this_worker = int(args.n_trials)
    else:
        trials_this_worker = 12  # default small micro-HPO

    # Create study
    if args.storage and args.study_name:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )
        print(f"[Optuna] using shared storage={args.storage} study={args.study_name}")
    elif args.storage or args.study_name:
        raise ValueError("If you specify --storage or --study_name, you must specify both.")
    else:
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
        print("[Optuna] in-memory study")

    # Objective
    def objective(trial: optuna.Trial) -> float:
        # Fixed knobs
        arch  = args.fixed_arch
        batch = int(args.fixed_batch)
        lr    = float(args.fixed_lr)

        # Search: weight decay
        wd = trial.suggest_float("weight_decay", float(args.wd_min), float(args.wd_max), log=True)
        # Search: scheduler
        sch = trial.suggest_categorical("scheduler", list(args.scheduler_choices))

        # Hyper-params per scheduler
        warmup_epochs = 0
        step_size = int(args.step_min)
        gamma = 1.0
        plateau_factor = float(args.plateau_factor_min)
        plateau_patience = int(args.plateau_patience_min)
        plateau_cooldown = int(args.plateau_cooldown_min)

        if sch == "cosine":
            warmup_pct = trial.suggest_float("warmup_pct",
                                             float(args.warmup_pct_min),
                                             float(args.warmup_pct_max))
            warmup_epochs = max(1, int(round(warmup_pct * args.epochs)))

        elif sch == "step":
            step_size = trial.suggest_int("step_size",
                                          int(args.step_min),
                                          int(args.step_max))
            gamma = trial.suggest_float("step_gamma",
                                        float(args.step_gamma_min),
                                        float(args.step_gamma_max))

        elif sch == "exp":
            gamma = trial.suggest_float("exp_gamma",
                                        float(args.exp_gamma_min),
                                        float(args.exp_gamma_max))

        elif sch == "plateau":
            plateau_factor = trial.suggest_float("plateau_factor",
                                                 float(args.plateau_factor_min),
                                                 float(args.plateau_factor_max),
                                                 log=True)
            plateau_patience = trial.suggest_int("plateau_patience",
                                                 int(args.plateau_patience_min),
                                                 int(args.plateau_patience_max))
            plateau_cooldown = trial.suggest_int("plateau_cooldown",
                                                 int(args.plateau_cooldown_min),
                                                 int(args.plateau_cooldown_max))
        else:
            # Should not happen due to choices
            pass

        # trial artifact dir (compact tag)
        tag = f"trial_{trial.number:03d}_{_safe_dirname(arch)}_bs{batch}_sch{sch}_wd{wd:.1e}"
        if sch == "cosine":
            tag += f"_wu{warmup_epochs}"
        elif sch == "step":
            tag += f"_s{step_size}_g{gamma:.2f}"
        elif sch == "exp":
            tag += f"_g{gamma:.3f}"
        elif sch == "plateau":
            tag += f"_f{plateau_factor:.2f}_p{plateau_patience}_c{plateau_cooldown}"

        save_dir = os.path.join(args.out_root, tag)
        os.makedirs(save_dir, exist_ok=True)

        # Build args for train.py
        clip_q_val = None if (args.clip_q is not None and args.clip_q < 0) else args.clip_q
        train_args = _mk_args_for_train(
            slsim_lenses=args.slsim_lenses,
            slsim_nonlenses=args.slsim_nonlenses,
            hsc_lenses=args.hsc_lenses,
            hsc_nonlenses=args.hsc_nonlenses,
            save_dir=save_dir,
            arch=arch,
            lr=lr,
            batch_size=batch,
            weight_decay=wd,
            scheduler=sch,
            warmup_epochs=warmup_epochs,
            min_lr=float(args.min_lr),
            step_size=int(step_size),
            gamma=float(gamma),
            plateau_factor=float(plateau_factor),
            plateau_patience=int(plateau_patience),
            plateau_cooldown=int(plateau_cooldown),
            plateau_monitor=str(args.plateau_monitor),
            drop_path=float(args.drop_path),
            epochs=int(args.epochs),
            patience=int(args.patience),
            seed=int(args.seed),
            device=str(args.device),
            num_workers=int(args.workers),
            take_train_frac=args.take_train_frac,
            take_val_fraction=args.take_val_fraction,
            take_test_fraction=args.take_test_fraction,
            apply_padding=bool(args.apply_padding),
            out_size_when_padded=int(args.out_size_when_padded),
            apply_normalization=bool(args.apply_normalization),
            clip_q=clip_q_val,
            low_clip_q=args.low_clip_q,
            use_mad=bool(args.use_mad),
            smoothing_mode=str(args.smoothing_mode),
            gaussian_sigma=float(args.gaussian_sigma),
            guided_radius=int(args.guided_radius),
            guided_eps=float(args.guided_eps),
        )

        # Run training
        t0 = time.time()
        train_mod.main(train_args)
        dur = int(time.time() - t0)

        # Parse best val AUC
        csv_path = os.path.join(save_dir, "training_log.csv")
        best_val_auc = _read_best_val_auc(csv_path)

        # Attach attrs for later analysis
        trial.set_user_attr("save_dir", save_dir)
        trial.set_user_attr("duration_sec", dur)

        # Trial summary line
        extra = ""
        if sch == "cosine":
            extra = f"wu={warmup_epochs}"
        elif sch == "step":
            extra = f"step={step_size} gamma={gamma:.3f}"
        elif sch == "exp":
            extra = f"gamma={gamma:.4f}"
        elif sch == "plateau":
            extra = f"factor={plateau_factor:.3f} patience={plateau_patience} cooldown={plateau_cooldown}"

        print(
            f"[trial {trial.number}] arch={arch} lr={lr:.2e} bs={batch} wd={wd:.2e} sch={sch} {extra} "
            f"| best_val_auc={best_val_auc:.6f} | {dur}s"
        )

        return float(best_val_auc)

    # Optimize
    study.optimize(objective, n_trials=trials_this_worker, gc_after_trial=True)

    # Summary (per worker)
    completed = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
    if len(completed) > 0:
        print("\n=== Worker summary ===")
        print(f"Best value:  {study.best_value:.6f}")
        print(f"Best params: {study.best_trial.params}")
        sd = study.best_trial.user_attrs.get("save_dir", None)
        if sd:
            print(f"Artifacts at: {sd}")
    else:
        print("\nNo completed trials in this worker.")

if __name__ == "__main__":
    main()
