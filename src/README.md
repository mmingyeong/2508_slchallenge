
Github Repo: https://github.com/mmingyeong/2508_slchallenge.git

# Strong Lensing Classification (ConvNeXt V2, 1-channel FITS)

This repository implements a binary classification pipeline for the **Strong Lens Data Challenge**, using ConvNeXt V2 models to classify strong gravitational lenses from real and simulated LSST-like data.

It includes training (`train.py`), inference (`predict.py`), evaluation (`evaluate.py`), and model definition (`model.py`) for binary classification on 41×41 single-band FITS cutouts.  
The architecture uses ConvNeXt V2 with grayscale input (`in_chans=1`) and a single-logit output (`num_classes=1`, BCEWithLogitsLoss).

---

## 1) Environment

* Python ≥ 3.10 (tested with 3.12)
* PyTorch ≥ 2.2 (CUDA 12.x)
* NumPy, pandas, scikit-learn, tqdm, timm, astropy

---

## 2) Data Layout

The pipeline expects four folders:

```

<slsim_lenses>/           <slsim_nonlenses>/
<hsc_lenses>/             <hsc_nonlenses>/

```

Each folder contains FITS files (label inferred by folder name).  
For **external test**, the same structure can be used with unlabeled FITS files or symlinks.

---

## 3) Final Submission

The final submission file follows the challenge format:

```

id,preds,RA,Dec,zlens,vel disp,ell_m,ell_m_PA,sh,sh_PA,Rein,ell_l,ell_l_PA,Reff_l,n_l_sers,mag_lens_g,mag_lens_r,mag_lens_i,mag_lens_z,mag_lens_y,zsrc,srcx,srcy,mag_src_g,mag_src_r,mag_src_i,mag_src_z,mag_src_y,ell_s,ell_s_PA,Reff_s,n_s_sers
OBJID_0001,1,-99,-99, ... (fill -99 for regression fields)
...

```

Only the classification (`preds`) column is used; all regression columns are filled with `-99`.

Final submission file:  
**`submission_nano_ensemble.csv`**

---

## 4) Final Submission Details

The **final challenge submission** was produced using the **ConvNeXtV2-Nano ensemble model**, averaging predictions from five independently trained seeds (`s42`, `s101`, `s202`, `s303`, `s404`).  
This Nano ensemble achieved the best validation and test performance among all configurations and was used to generate the official results.

---

## 5) Reproducibility Notes

* **Model:** `model.py` — ConvNeXt V2 blocks with GRN, DropPath, grayscale input, and single-logit output.  
* **Training:** `train.py` — supports multiple schedulers, early stopping, AMP, and flexible preprocessing.  
* **Inference:** `predict.py` — identical preprocessing to training; saves predictions and probabilities per split.

---

**Author:** Mingyeong Yang (KASI / UST)  
**Date:** October 2025  
**Email:** mmingyeong@kasi.re.kr  
**License:** Research / Academic use only.
```

---