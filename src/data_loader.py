# data_loader.py
# -*- coding: utf-8 -*-
"""
Strong-lensing binary classification loader with independent toggles:
- apply_padding: center constant-pad 41x41 -> out_size_when_padded (default 64)
- apply_normalization: background subtraction -> (optional) high-quantile clip -> z-score (or MAD)
- smoothing: one of {none, gaussian, guided, psf} applied BEFORE normalization and BEFORE padding
...
- Smoothing is applied on the original 41x41 grid (not on padded images), BEFORE normalization.
"""


from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import logging, warnings, random, math

import numpy as np
import torch
import torch.nn.functional as F
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


# ---------------- helper: safe collate factory (filters None) ----------------
def make_safe_collate(out_size: int) -> Callable:
    """
    Returns a collate_fn that filters None samples and emits an empty batch
    with shape (0,1,out_size,out_size) if needed.
    """
    def safe_collate(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return (
                torch.empty(0, 1, out_size, out_size, dtype=torch.float32),
                torch.empty(0, dtype=torch.long),
                [],
            )
        xs, ys, metas = zip(*batch)
        return torch.stack(xs, 0), torch.stack(ys, 0), list(metas)
    return safe_collate


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
    with a vector column of length 1681.
    """
    try:
        if _USE_FITSIO:
            with fitsio.FITS(fits_path) as f:
                data = None
                # 1) try PRIMARY
                try:
                    arr = f[0].read()
                    if isinstance(arr, np.ndarray) and arr.size == 41*41:
                        data = arr.reshape(41, 41)
                except Exception:
                    data = None
                # 2) fallback: BinTable HDU(1)
                if data is None and len(f) > 1 and f[1].get_type() == "BINARY_TBL":
                    tbl = f[1].read()
                    names = list(tbl.dtype.names or [])
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
        else:
            with fits.open(fits_path, memmap=False) as hdul:
                data = None
                # 1) try PRIMARY
                arr0 = hdul[0].data
                if isinstance(arr0, np.ndarray) and arr0.size == 41*41:
                    data = arr0.reshape(41, 41)
                # 2) fallback: BinTable HDU(1)
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
            return None
        return torch.from_numpy(data)
    except Exception:
        return None


# ---------------- smoothing utilities ----------------

def _gaussian_kernel1d(sigma: float, truncate: float = 3.0) -> torch.Tensor:
    """Return 1D Gaussian kernel normalized to sum=1 (dtype=float32)."""
    sigma = float(max(sigma, 1e-6))
    radius = int(math.ceil(truncate * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    weights = torch.exp(-0.5 * (x / sigma) ** 2)
    weights /= weights.sum().clamp_min(1e-12)
    return weights  # (K,)

def _gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Separable Gaussian blur for a single-channel 2D tensor x (H,W) -> (H,W).
    """
    if sigma <= 0:
        return x
    k1d = _gaussian_kernel1d(sigma)  # (K,)
    Kh = k1d.view(1, 1, 1, -1)       # horizontal
    Kv = k1d.view(1, 1, -1, 1)       # vertical
    x4 = x.unsqueeze(0).unsqueeze(0) # (1,1,H,W)
    xh = F.conv2d(x4, Kh, padding=(0, Kh.shape[-1]//2))
    xv = F.conv2d(xh, Kv, padding=(Kv.shape[-2]//2, 0))
    return xv.squeeze(0).squeeze(0)

def _box_filter(x4: torch.Tensor, r: int) -> torch.Tensor:
    """
    Box filter by conv2d (batch,1,H,W) with kernel size (2r+1).
    Returns same shape as x4. No normalization (raw sum).
    """
    k = 2 * r + 1
    weight = torch.ones((1, 1, k, k), dtype=x4.dtype, device=x4.device)
    return F.conv2d(x4, weight, padding=r)

def _guided_filter_self(x: torch.Tensor, r: int, eps: float) -> torch.Tensor:
    """
    Self-guided filter (gray-scale). x: (H,W) float tensor.
    Implements He et al. guided filter with I = p = x.
    """
    H, W = x.shape
    x4 = x.view(1, 1, H, W)
    N = _box_filter(torch.ones_like(x4), r)  # window size per pixel

    mean_x = _box_filter(x4, r) / N
    mean_xx = _box_filter(x4 * x4, r) / N
    var_x = (mean_xx - mean_x * mean_x).clamp_min(0.0)

    a = var_x / (var_x + eps)
    b = mean_x - a * mean_x

    mean_a = _box_filter(a, r) / N
    mean_b = _box_filter(b, r) / N

    q = mean_a * x4 + mean_b
    return q.view(H, W)

def _fwhm_to_sigma(fwhm: float) -> float:
    return float(fwhm) / 2.354820045  # 2*sqrt(2*ln2)

def _psf_sigma_kernel(current_sigma: float, target_sigma: float) -> float:
    """
    Given current PSF sigma and target PSF sigma (both in pixels),
    return required Gaussian kernel sigma to homogenize: sigma_k = sqrt(target^2 - current^2),
    clipped at 0 if target <= current.
    """
    if target_sigma <= current_sigma:
        return 0.0
    return float(math.sqrt(max(target_sigma**2 - current_sigma**2, 0.0)))


# ---------------- Dataset ----------------
class LensFITSBinaryDataset(Dataset):
    def __init__(
        self,
        files: List[str],
        labels: List[int],
        domains: List[str],
        augment: bool = False,
        eps: float = 1e-6,
        # independent toggles
        apply_padding: bool = False,
        out_size_when_padded: int = 64,
        apply_normalization: bool = True,
        clip_q: Optional[float] = 0.997,       # None -> no clipping, just z-score
        low_clip_q: Optional[float] = None,    # e.g., 0.005 if needed
        use_mad: bool = False,                 # robust normalization via median/MAD
        # smoothing
        smoothing_mode: str = "none",          # {"none","gaussian","guided","psf"}
        gaussian_sigma: float = 0.8,           # in pixels on 41x41
        guided_radius: int = 2,                # window radius r (window size = 2r+1)
        guided_eps: float = 1e-2,              # regularization (in image intensity^2)
        psf_target_fwhm: Optional[float] = None,               # in pixels
        psf_input_fwhm_by_domain: Optional[Dict[str, float]] = None,  # {"hsc":..., "slsim":...}
    ):
        """
        apply_padding:
            True  -> constant-pad to out_size_when_padded (default 64)
            False -> keep 41x41
        apply_normalization:
            True  -> background subtraction -> (optional) high-quantile clip -> z-score (or MAD)
            False -> raw values (float32), no normalization
        smoothing:
            Applied AFTER normalization, BEFORE padding.
            - "none": no smoothing
            - "gaussian": Gaussian blur with gaussian_sigma (pixels)
            - "guided": self-guided filter with guided_radius r and guided_eps
            - "psf": Gaussian convolution to reach psf_target_fwhm per domain
        """
        assert len(files) == len(labels) == len(domains)
        self.files = files
        self.labels = labels
        self.domains = domains
        self.augment = augment
        self.eps = eps

        self.apply_padding = apply_padding
        self.out_size_when_padded = out_size_when_padded
        self.apply_normalization = apply_normalization
        self.clip_q = clip_q
        self.low_clip_q = low_clip_q
        self.use_mad = use_mad

        # smoothing params
        self.smoothing_mode = (smoothing_mode or "none").lower()
        assert self.smoothing_mode in {"none", "gaussian", "guided", "psf"}
        self.gaussian_sigma = float(gaussian_sigma)
        self.guided_radius = int(guided_radius)
        self.guided_eps = float(guided_eps)
        self.psf_target_fwhm = psf_target_fwhm
        self.psf_input_fwhm_by_domain = psf_input_fwhm_by_domain or {}

        n_lens = int(np.sum(labels))
        out_h = out_size_when_padded if apply_padding else 41
        logger.info(
            "Dataset: N=%d | lens=%d | nonlens=%d | augment=%s | padding=%s(out=%d) | "
            "normalization=%s | clip_q=%s | smoothing=%s",
            len(files), n_lens, len(files)-n_lens, augment,
            apply_padding, out_h, apply_normalization, str(clip_q), self.smoothing_mode
        )

    def __len__(self): return len(self.files)

    @staticmethod
    def _augment(img: torch.Tensor) -> torch.Tensor:
        # 1. 기본 대칭 변환 (현재 유지)
        k = random.randint(0, 3)
        img = torch.rot90(img, k, dims=(-2, -1))
        if random.random() < 0.5: img = torch.flip(img, dims=[-1])
        if random.random() < 0.5: img = torch.flip(img, dims=[-2])

        # 2. 현실적인 노이즈/밝기 변환 추가
        if random.random() < 0.5:
            # 가우시안 노이즈 추가 (데이터셋의 통계에 맞게 std 조정 필요)
            noise = torch.randn_like(img) * 0.01 
            img = img + noise

        if random.random() < 0.5:
            # 밝기 및 대비 조절 (예: 무작위 감마 보정)
            gamma = random.uniform(0.8, 1.2)
            img = torch.pow(img.clamp(min=1e-6), gamma) # clamp로 0 나누기 방지

        return img
    
    def _normalize(self, img: torch.Tensor) -> torch.Tensor:
        """
        Background subtraction -> optional high-quantile clip -> z-score (or MAD).
        Operates on 41x41; smoothing (if any) is applied beforehand.
        """


        x = img.clone()

        # (1) Background subtraction: median of lowest 20% pixels
        v = x.flatten()
        q20 = torch.quantile(v, 0.20)
        low = v[v <= q20]
        bkg = torch.median(low) if low.numel() > 0 else torch.median(v)
        x = x - bkg

        # (2) High-tail clipping (and optional low-tail)
        if self.clip_q is not None:
            v = x.flatten()
            hi = torch.quantile(v, self.clip_q)
            if self.low_clip_q is not None:
                lo = torch.quantile(v, self.low_clip_q)
                x = torch.clamp(x, min=lo, max=hi)
            else:
                x = torch.clamp_max(x, hi)

        # (3) z-score (or MAD) normalization
        if self.use_mad:
            med = torch.median(x)
            mad = torch.median(torch.abs(x - med))
            scale = 1.4826 * mad + self.eps
            x = (x - med) / scale
        else:
            m = torch.mean(x)
            s = torch.std(x, unbiased=False) + self.eps
            x = (x - m) / s

        return x

    def _apply_smoothing(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        """
        Apply smoothing on (41,41) tensor AFTER normalization, BEFORE padding.
        """
        mode = self.smoothing_mode
        if mode == "none":
            return x

        if mode == "gaussian":
            return _gaussian_blur_2d(x, self.gaussian_sigma)

        if mode == "guided":
            r = max(1, self.guided_radius)
            eps = max(1e-8, self.guided_eps)
            return _guided_filter_self(x, r=r, eps=eps)

        if mode == "psf":
            # require target fwhm and per-domain input fwhm
            if self.psf_target_fwhm is None:
                return x
            target_sigma = _fwhm_to_sigma(self.psf_target_fwhm)
            current_fwhm = self.psf_input_fwhm_by_domain.get(str(domain).lower(), None)
            if current_fwhm is None:
                # if missing domain info, skip (no blur) to avoid over-smoothing
                return x
            current_sigma = _fwhm_to_sigma(current_fwhm)
            sigma_k = _psf_sigma_kernel(current_sigma, target_sigma)
            if sigma_k <= 0:
                return x
            return _gaussian_blur_2d(x, sigma_k)

        # fallback
        return x

    def _maybe_pad(self, x: torch.Tensor, bg_value: float = 0.0) -> torch.Tensor:
        if not self.apply_padding:
            return x
        H, W = x.shape[-2], x.shape[-1]
        out_size = self.out_size_when_padded
        dh, dw = out_size - H, out_size - W
        if dh < 0 or dw < 0:
            raise ValueError(f"out_size_when_padded={out_size} smaller than input ({H},{W})")
        pt, pb = dh // 2, dh - dh // 2
        pl, pr = dw // 2, dw - dw // 2
        x = F.pad(x.unsqueeze(0), (pl, pr, pt, pb), mode="constant", value=bg_value).squeeze(0)
        return x

    def __getitem__(self, idx: int):
        fp = self.files[idx]
        y = int(self.labels[idx])
        domain = self.domains[idx]
        try:
            img = _read_fits_image_41x41(fp)  # (41,41)
            if img is None or (not torch.isfinite(img).all()):
                raise ValueError(f"Skipping corrupted file: {fp}")

            # start from raw float
            x = img.to(torch.float32)          # (41,41) raw

            # 1) smoothing (BEFORE normalization)
            x = self._apply_smoothing(x, domain=domain)

            # 2) normalization
            if self.apply_normalization:
                x = self._normalize(x)         # normalize the smoothed image


            # padding (after smoothing)
            x = self._maybe_pad(x)             # (41,41) or (64,64)
            x = x.unsqueeze(0).to(torch.float32)

            # optional augmentation
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
    augment_train: bool = False,
    pin_memory: bool = True,
    take_train_fraction: Optional[float] = None,
    take_val_fraction: Optional[float] = None,
    take_test_fraction: Optional[float] = None,
    # toggles & knobs (padding/normalization)
    apply_padding: bool = False,
    out_size_when_padded: int = 64,
    apply_normalization: bool = True,
    clip_q: Optional[float] = 0.997,
    low_clip_q: Optional[float] = None,
    use_mad: bool = False,
    # smoothing knobs
    smoothing_mode: str = "none",            # {"none","gaussian","guided","psf"}
    gaussian_sigma: float = 0.8,
    guided_radius: int = 2,
    guided_eps: float = 1e-2,
    psf_target_fwhm: Optional[float] = None,
    psf_input_fwhm_by_domain: Optional[Dict[str, float]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    IMPORTANT: For fair evaluation, keep the same toggles for train/val/test.
    """
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
    if take_train_fraction and 0 < take_train_fraction < 1.0:
        k = max(1, int(len(tr_f) * take_train_fraction))
        keep = rng.choice(len(tr_f), size=k, replace=False)
        tr_f, tr_y, tr_d = [tr_f[i] for i in keep], [tr_y[i] for i in keep], [tr_d[i] for i in keep]
        logger.info(f"Train subsampling: kept {k}/{len(tr_idx)} ({take_train_fraction*100:.2f}%)")
    if take_val_fraction and 0 < take_val_fraction < 1.0:
        k = max(1, int(len(va_f) * take_val_fraction))
        keep = rng.choice(len(va_f), size=k, replace=False)
        va_f, va_y, va_d = [va_f[i] for i in keep], [va_y[i] for i in keep], [va_d[i] for i in keep]
        logger.info(f"Val subsampling: kept {k}/{len(va_idx)} ({take_val_fraction*100:.2f}%)")
    if take_test_fraction and 0 < take_test_fraction < 1.0:
        k = max(1, int(len(te_f) * take_test_fraction))
        keep = rng.choice(len(te_f), size=k, replace=False)
        te_f, te_y, te_d = [te_f[i] for i in keep], [te_y[i] for i in keep], [te_d[i] for i in keep]
        logger.info(f"Test subsampling: kept {k}/{len(te_idx)} ({take_test_fraction*100:.2f}%)")

    # --- build datasets (avoid duplicate 'augment' kw) ---
    ds_args_common = dict(
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
        psf_target_fwhm=psf_target_fwhm,
        psf_input_fwhm_by_domain=psf_input_fwhm_by_domain,
    )

    ds_tr = LensFITSBinaryDataset(
        tr_f, tr_y, tr_d,
        augment=augment_train,
        **ds_args_common
    )
    ds_va = LensFITSBinaryDataset(
        va_f, va_y, va_d,
        augment=False,
        **ds_args_common
    )
    ds_te = LensFITSBinaryDataset(
        te_f, te_y, te_d,
        augment=False,
        **ds_args_common
    )

    out_size = out_size_when_padded if apply_padding else 41
    collate_fn = make_safe_collate(out_size)

    logger.info(
        "Split | Train=%d | Val=%d | Test=%d | batch=%d | out_size=%d | smoothing=%s",
        len(ds_tr), len(ds_va), len(ds_te), batch_size, out_size, smoothing_mode
    )

    train_loader = DataLoader(
        ds_tr, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_fn, persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        ds_va, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_fn, persistent_workers=(num_workers > 0)
    )
    test_loader = DataLoader(
        ds_te, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_fn, persistent_workers=(num_workers > 0)
    )
    return train_loader, val_loader, test_loader
