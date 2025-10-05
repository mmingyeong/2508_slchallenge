Github Repo: https://github.com/mmingyeong/2508_slchallenge.git

# Strong Lensing Classification (ConvNeXt V2, 1-channel FITS)

This repository implements a binary classification pipeline for the Strong Lens Data Challenge, using ConvNeXt V2 models to classify strong gravitational lenses from real and simulated LSST-like data.

This repository contains training (`train.py`), inference (`predict.py`), evaluation (`evaluate.py`), and model definition (`model.py`) for binary strong-lensing classification on 41×41 single-band FITS cutouts.
Architecture is ConvNeXt V2 with grayscale input (`in_chans=1`) and a 1-logit head (`num_classes=1`, BCEWithLogitsLoss).

## 1) Environment

* Python ≥ 3.10 (tested with 3.12)
* PyTorch ≥ 2.2, CUDA 12.x
* NumPy, pandas, scikit-learn, tqdm, timm, astropy (for FITS I/O in the dataloader)


## 2) Data layout expected by the code

The pipeline expects **four folders** (any content; used only as *buckets*):

```
<slsim_lenses>/           <slsim_nonlenses>/
<hsc_lenses>/             <hsc_nonlenses>/
```

Each folder contains FITS files; labels are implied by the folder name.
During *external test inference*, the same four folders can simply hold **symlinks** to the unlabeled FITS so that `predict.py` can iterate the files without caring about labels.

## 3) Training (single model)

`train.py` builds ConvNeXt V2 (atto/nano/tiny), trains with BCEWithLogitsLoss, and supports common preprocessing toggles (padding / normalization / optional smoothing).

Example (Nano, seed 42):

```bash
python src/train.py \
  --slsim_lenses   /path/to/slsim_lenses \
  --slsim_nonlenses /path/to/slsim_nonlenses \
  --hsc_lenses     /path/to/hsc_lenses \
  --hsc_nonlenses  /path/to/hsc_nonlenses \
  --model_size nano --seed 42 \
  --epochs 20 --batch_size 128 --num_workers 8 \
  --apply_normalization --smoothing_mode gaussian --gaussian_sigma 1.0 \
  --drop_path 0.05 --device cuda \
  --save_dir ./res/_full_ckpt_fulltrain_nano_s42_YYYYMMDD_HHMMSS
```

Artifacts:

* `best.pt`, `last.pt`
* `training_log.csv`, `train.log`
* `results.json` (final test metrics for the split inside the loader)

Train multiple seeds (e.g., 42/101/202/303/404) to ensemble later.

## 4) Validation ensemble & threshold (t*)

After each seed makes **VAL predictions** via `predict.py` (below), ensemble their `pred_val.csv` (path-wise average of probabilities) and **optimize threshold t*** using Youden’s J on VAL. Any equivalent script is fine; our typical flow:

```bash
# Example: run predict on VAL for each seed (output pred_val.csv per seed)
python src/predict.py ... --which val --model_path <ckpt>/best.pt --model_size nano --output_dir <ckpt>_pred_val_YYYYMMDD_HHMMSS --apply_normalization --smoothing_mode gaussian --gaussian_sigma 1.0

# Then ensemble VAL CSVs and optimize t*
# (you can reuse your existing sl_eval_threshold_val_ens*.pbs or an equivalent script)
```

Result:

* `val_ens.csv`
* `metrics_val_ens*.json` with `global.threshold` (= t*)
* `threshold.txt` (optional convenience)

## 5) Test inference (per seed)

Run `predict.py` on the **test split**. The script iterates the four folders; for unlabeled external test, populate them with **symlinks** to all FITS files.

Example:

```bash
python src/predict.py \
  --slsim_lenses   /test_buckets/slsim_lenses \
  --slsim_nonlenses /test_buckets/slsim_nonlenses \
  --hsc_lenses     /test_buckets/hsc_lenses \
  --hsc_nonlenses  /test_buckets/hsc_nonlenses \
  --which test \
  --batch_size 128 --num_workers 8 \
  --model_path ./res/_full_ckpt_fulltrain_nano_s42_.../best.pt \
  --model_size nano --drop_path 0.05 --device cuda \
  --output_dir ./res/_full_ckpt_fulltrain_nano_s42_..._pred_test_YYYYMMDD_HHMMSS \
  --apply_normalization --smoothing_mode gaussian --gaussian_sigma 1.0
```

Output per seed:

* `pred_test.csv` with columns: `path, domain, label(dummy), prob, pred`

## 6) Test ensemble + fixed threshold

Merge all seeds’ `pred_test.csv` by filename, average `prob`, and apply the **VAL-derived t***:

```text
pred = 1 if prob_mean >= t* else 0
```

Save `test_ens_labeled.csv` (if labels exist) or `test_ens.csv` (unlabeled).

## 7) Final submission file

Competition requires:

* **Required columns**: `id, preds`
* **Regression columns**: fill with `-99` if not submitting (RA, Dec, zlens, …)

Create `submission.csv` with:

```
id,preds,RA,Dec,zlens,vel disp,ell_m,ell_m_PA,sh,sh_PA,Rein,ell_l,ell_l_PA,Reff_l,n_l_sers,mag_lens_g,mag_lens_r,mag_lens_i,mag_lens_z,mag_lens_y,zsrc,srcx,srcy,mag_src_g,mag_src_r,mag_src_i,mag_src_z,mag_src_y,ell_s,ell_s_PA,Reff_s,n_s_sers
OBJID_0001,1,-99,-99, ... (fill -99 for all regression fields)
...
```

Where `id` is the FITS stem (filename without extension).

## 8) Reproducibility notes

* **Model**: `model.py` implements ConvNeXt V2 blocks with GRN and DropPath; grayscale input; 1-logit head.
* **Training**: `train.py` supports cosine/step/exp/plateau schedulers, early stopping, AMP, and selectable monitor (`loss` or `auc`).
* **Inference**: `predict.py` mirrors the preprocessing toggles; saves CSV/NPY per split.

## 9) Minimal run sheet (Nano 5-seed ensemble)

1. Train seeds: `--model_size nano --seed 42/101/202/303/404`
2. Predict VAL per seed → get `pred_val.csv` per seed
3. Ensemble VAL & optimize t* (store `threshold.txt`)
4. Predict TEST per seed → `pred_test.csv` per seed
5. Ensemble TEST (mean prob) + apply t* → `test_ens.csv`
6. Convert to submission format → `submission_nano_ensemble.csv`

---

**Contacts / Owner:**
Mingyeong Yang (KASI / UST) — October 2025 mmingyeong@kasi.re.kr

**License:** research/academic use.
