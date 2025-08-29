# data_loader.py
# -*- coding: utf-8 -*-
"""
Very simple data loader for strong-lensing binary classification (41x41).
- Reads FITS BinTable: HDU1 has two cols [meta(str), image(1681)]
- Returns (x: (1,41,41) float32 z-score, y: {0,1}, meta: {'path','domain'})
- Corrupted FITS -> skip (None) and filtered by safe_collate
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging, warnings, random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# try fitsio, else astropy
try:
    import fitsio  # type: ignore
    _USE_FITSIO = True
except Exception:
    from astropy.io import fits  # type: ignore
    _USE_FITSIO = False

# tidy logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [data_loader] %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)
logger = logging.getLogger("data_loader")

# reduce noisy FITS warnings
try:
    from astropy.io.fits.verify import VerifyWarning  # type: ignore
    warnings.filterwarnings("ignore", category=VerifyWarning)
    warnings.filterwarnings("ignore", message="File may have been truncated")
except Exception:
    pass


# ---------------- helper: safe collate (filters None) ----------------
def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return (
            torch.empty(0, 1, 41, 41, dtype=torch.float32),
            torch.empty(0, dtype=torch.long),
            [],
        )
    xs, ys, metas = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0), list(metas)


# ---------------- helper: collect files ----------------
def collect_files(class_paths: Dict[str, str]) -> Tuple[List[str], List[int], List[str]]:
    files, labels, domains = [], [], []
    for key, d in class_paths.items():
        dpath = Path(d)
        if not dpath.exists():
            logger.warning(f"Missing directory: {dpath}")
            continue
        label = 0 if "nonlenses" in key.lower() else 1
        domain = "slsim" if "slsim" in key.lower() else "hsc"
        fs = sorted(str(p) for p in dpath.glob("*.fits"))
        files.extend(fs)
        labels.extend([label] * len(fs))
        domains.extend([domain] * len(fs))
        logger.info(f"Collected {len(fs)} from '{key}' ({dpath}), label={label}, domain={domain}")
    logger.info(f"TOTAL files collected: {len(files)}")
    return files, labels, domains


# ---------------- helper: read 41x41 image ----------------
def _read_fits_image_41x41(fits_path: str):
    """
    Return torch.FloatTensor (41,41) or None.
    Supports files where PRIMARY is empty and the image lives in HDU 1 BinTable
    with two columns [meta(str), image(1681D)].
    """
    try:
        if _USE_FITSIO:
            import fitsio
            with fitsio.FITS(fits_path) as f:
                data = None
                # 1) try PRIMARY image (rare in this dataset)
                try:
                    arr = f[0].read()
                    if isinstance(arr, np.ndarray) and arr.size == 41*41:
                        data = arr.reshape(41, 41)
                except Exception:
                    data = None
                # 2) fallback: BinTable HDU (1): pick the column of length 1681
                if data is None and len(f) > 1 and f[1].get_type() == "BINARY_TBL":
                    tbl = f[1].read()  # structured array with 1 row x 2 cols
                    names = list(tbl.dtype.names or [])
                    col_idx = None
                    # prefer a column literally named 'image'
                    for i, n in enumerate(names):
                        if str(n).lower() == "image":
                            col_idx = i
                            break
                    # else find the first vector column of length 1681
                    if col_idx is None:
                        for i, n in enumerate(names):
                            v = np.asarray(tbl[n][0])
                            if v.size == 41*41:
                                col_idx = i
                                break
                    if col_idx is not None:
                        v = np.asarray(tbl[names[col_idx]][0])
                        data = v.reshape(41, 41)
        else:
            from astropy.io import fits
            with fits.open(fits_path, memmap=False) as hdul:
                data = None
                # 1) try PRIMARY
                arr0 = hdul[0].data
                if isinstance(arr0, np.ndarray) and arr0.size == 41*41:
                    data = arr0.reshape(41, 41)
                # 2) fallback: BinTable HDU (1)
                if data is None and len(hdul) > 1 and isinstance(hdul[1], fits.BinTableHDU):
                    tbl = hdul[1].data
                    names = list(tbl.columns.names or [])
                    col_idx = None
                    for i, n in enumerate(names):
                        if str(n).lower() == "image":
                            col_idx = i
                            break
                    if col_idx is None:
                        for i, n in enumerate(names):
                            v = np.asarray(tbl[n][0])
                            if v.size == 41*41:
                                col_idx = i
                                break
                    if col_idx is not None:
                        v = np.asarray(tbl[names[col_idx]][0])
                        data = v.reshape(41, 41)

        if data is None:
            return None
        data = np.asarray(data, dtype=np.float32)
        if not np.isfinite(data).all():
            return None  # guard against NaN/Inf
        return torch.from_numpy(data)
    except Exception:
        return None


# ---------------- Dataset ----------------
class LensFITSBinaryDataset(Dataset):
    def __init__(
        self,
        files: List[str],
        labels: List[int],
        domains: List[str],
        augment: bool = False,
        eps: float = 1e-6,
    ):
        assert len(files) == len(labels) == len(domains)
        self.files = files
        self.labels = labels
        self.domains = domains
        self.augment = augment
        self.eps = eps
        n_lens = int(np.sum(labels))
        logger.info(f"Dataset: N={len(files)} | lens={n_lens} | nonlens={len(files)-n_lens} | augment={augment}")

    def __len__(self): return len(self.files)

    @staticmethod
    def _augment(img: torch.Tensor) -> torch.Tensor:
        k = random.randint(0, 3)
        img = torch.rot90(img, k, dims=(-2, -1))
        if random.random() < 0.5: img = torch.flip(img, dims=[-1])
        if random.random() < 0.5: img = torch.flip(img, dims=[-2])
        return img

    def __getitem__(self, idx: int):
        fp = self.files[idx]
        y = int(self.labels[idx])
        domain = self.domains[idx]
        try:
            img = _read_fits_image_41x41(fp)               # (41,41)

            if img is None:
                # You can either raise an error, or, if you want to skip,
                # you might need a different strategy for the DataLoader.
                # For example, you could loop until you find a valid sample.
                # This requires more complex logic to prevent infinite loops.
                # A simpler way is to handle the corrupted files before training.
                # For now, let's just raise an error to stop execution and investigate.
                raise ValueError(f"Skipping corrupted file: {fp}")
            m, s = float(img.mean()), float(img.std()) + self.eps
            img = (img - m) / s
            x = img.unsqueeze(0).to(torch.float32)
            if self.augment:
                x = self._augment(x)
            return x, torch.tensor(y, dtype=torch.long), {"path": fp, "domain": domain}
        except Exception as e:
            logger.warning(f"Skip corrupt FITS: {fp} ({e})")
            return None


# ---------------- Public API ----------------
def get_dataloaders(
    class_paths: Dict[str, str],
    batch_size: int = 128,
    split: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    num_workers: int = 8,
    augment_train: bool = True,
    pin_memory: bool = True,
    take_train_fraction: Optional[float] = None,
    take_val_fraction: Optional[float] = None,
    take_test_fraction: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    assert abs(sum(split) - 1.0) < 1e-6, "split must sum to 1.0"

    files, labels, domains = collect_files(class_paths)

    idx_all = np.arange(len(files))
    tr_idx, tmp_idx, y_tr, y_tmp = train_test_split(
        idx_all, labels, test_size=(1.0 - split[0]), random_state=seed, stratify=labels
    )
    val_frac = split[1] / (split[1] + split[2] + 1e-12)
    va_idx, te_idx, _, _ = train_test_split(
        tmp_idx, y_tmp, test_size=(1.0 - val_frac), random_state=seed, stratify=y_tmp
    )

    def pick(idxs):
        return [files[i] for i in idxs], [int(labels[i]) for i in idxs], [domains[i] for i in idxs]

    tr_f, tr_y, tr_d = pick(tr_idx)
    va_f, va_y, va_d = pick(va_idx)
    te_f, te_y, te_d = pick(te_idx)

    rng = np.random.default_rng(seed)

    # --- train sampling ---
    if take_train_fraction and 0 < take_train_fraction < 1.0:
        k = max(1, int(len(tr_f) * take_train_fraction))
        keep = rng.choice(len(tr_f), size=k, replace=False)
        tr_f = [tr_f[i] for i in keep]
        tr_y = [tr_y[i] for i in keep]
        tr_d = [tr_d[i] for i in keep]
        logger.info(f"Train subsampling: kept {k}/{len(tr_idx)} ({take_train_fraction*100:.2f}%)")

    # --- val sampling ---
    if take_val_fraction and 0 < take_val_fraction < 1.0:
        k = max(1, int(len(va_f) * take_val_fraction))
        keep = rng.choice(len(va_f), size=k, replace=False)
        va_f = [va_f[i] for i in keep]
        va_y = [va_y[i] for i in keep]
        va_d = [va_d[i] for i in keep]
        logger.info(f"Val subsampling: kept {k}/{len(va_idx)} ({take_val_fraction*100:.2f}%)")

    # --- test sampling ---
    if take_test_fraction and 0 < take_test_fraction < 1.0:
        k = max(1, int(len(te_f) * take_test_fraction))
        keep = rng.choice(len(te_f), size=k, replace=False)
        te_f = [te_f[i] for i in keep]
        te_y = [te_y[i] for i in keep]
        te_d = [te_d[i] for i in keep]
        logger.info(f"Test subsampling: kept {k}/{len(te_idx)} ({take_test_fraction*100:.2f}%)")

    ds_tr = LensFITSBinaryDataset(tr_f, tr_y, tr_d, augment=augment_train)
    ds_va = LensFITSBinaryDataset(va_f, va_y, va_d, augment=False)
    ds_te = LensFITSBinaryDataset(te_f, te_y, te_d, augment=False)

    logger.info(f"Split | Train={len(ds_tr)} | Val={len(ds_va)} | Test={len(ds_te)} | batch={batch_size}")

    train_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              collate_fn=safe_collate, persistent_workers=(num_workers > 0))
    val_loader   = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              collate_fn=safe_collate, persistent_workers=(num_workers > 0))
    test_loader  = DataLoader(ds_te, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              collate_fn=safe_collate, persistent_workers=(num_workers > 0))
    return train_loader, val_loader, test_loader
