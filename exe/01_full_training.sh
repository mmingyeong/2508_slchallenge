#!/bin/bash
#PBS -N sl_fulltrain_atto_exp_ms
#PBS -q long
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l mem=48gb
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -M mmingyeong@kasi.re.kr
#PBS -o sl_fulltrain_atto_exp_ms.$PBS_ARRAYID.out
# >>> 멀티-시드 배열 잡(5개 시드) <<<
# 예: 5개 시드 중 동시에 최대 2개만 실행
#PBS -t 0-4%2


# ----------------------------
# User config
# ----------------------------
PROJECT_ROOT="/caefs/user/mmingyeong/2508_slchallence"
PYENV="py312"

# Data
SLSIM_LENSES="/caefs/data/IllustrisTNG/slchallenge/slsim_lenses/slsim_lenses"
SLSIM_NONLENSES="/caefs/data/IllustrisTNG/slchallenge/slsim_nonlenses/slsim_nonlenses"
HSC_LENSES="/caefs/data/IllustrisTNG/slchallenge/hsc_lenses/hsc_lenses"
HSC_NONLENSES="/caefs/data/IllustrisTNG/slchallenge/hsc_nonlenses/hsc_nonlenses"

# Training hyperparameters (final choice)
MODEL_SIZE="atto"
BATCH=128
LR=3.5e-4
WEIGHT_DECAY=1e-4
DROP_PATH=0.05
EPOCHS=80               # full training length (with early stopping)
PATIENCE=20             # monitored metric 기준
NUM_WORKERS=8

# Seeds for multi-run (mapped by PBS_ARRAYID)
SEEDS=(42 101 202 303 404)
IDX=${PBS_ARRAYID:-0}
if [[ $IDX -lt 0 || $IDX -ge ${#SEEDS[@]} ]]; then
  echo "Invalid PBS_ARRAYID=$IDX (must be 0..$(( ${#SEEDS[@]} - 1 )))"
  exit 1
fi
SEED=${SEEDS[$IDX]}

# Exponential scheduler: LR_T = LR0 * gamma^EPOCHS = LR0 * FINAL_LR_FACTOR
FINAL_LR_FACTOR=0.10
GAMMA=$(awk -v f=${FINAL_LR_FACTOR} -v e=${EPOCHS} 'BEGIN{printf("%.6f", exp(log(f)/e))}')

# Output
RUN_TAG="fulltrain_s${SEED}_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="${PROJECT_ROOT}/res/_full_ckpt_${RUN_TAG}"

# ----------------------------
# Environment
# ----------------------------
source ~/.bashrc
conda activate "${PYENV}"

echo "HOST    : $(hostname)"
echo "DATE    : $(date)"
echo "CUDA_DEV: ${CUDA_VISIBLE_DEVICES:-all}"
echo "SEED    : ${SEED}  (PBS_ARRAYID=${IDX})"
python -V || true
python - <<'PY' || true
import torch
print("PyTorch :", torch.__version__)
print("CUDA    :", torch.version.cuda)
print("Is CUDA :", torch.cuda.is_available())
PY
nvidia-smi || true

mkdir -p "${SAVE_DIR}"
echo "Computed gamma for ExponentialLR: ${GAMMA} (final LR ≈ ${FINAL_LR_FACTOR}×LR0 after ${EPOCHS} epochs)"

# ----------------------------
# Run (full data; augmentation enabled by default)
# Preprocessing: normalization + Gaussian smoothing (σ=1.0)
# Monitor: AUC for best checkpoint & early stopping
# ----------------------------
python -u "${PROJECT_ROOT}/src/train.py" \
  --slsim_lenses      "${SLSIM_LENSES}" \
  --slsim_nonlenses   "${SLSIM_NONLENSES}" \
  --hsc_lenses        "${HSC_LENSES}" \
  --hsc_nonlenses     "${HSC_NONLENSES}" \
  \
  --batch_size        ${BATCH} \
  --num_workers       ${NUM_WORKERS} \
  --train_frac        0.70 \
  --val_frac          0.15 \
  --test_frac         0.15 \
  \
  --model_size        "${MODEL_SIZE}" \
  --drop_path         ${DROP_PATH} \
  --lr                ${LR} \
  --weight_decay      ${WEIGHT_DECAY} \
  \
  --epochs            ${EPOCHS} \
  --patience          ${PATIENCE} \
  --min_delta         0.0 \
  --seed              ${SEED} \
  --device            cuda \
  --save_dir          "${SAVE_DIR}" \
  \
  --scheduler         exp \
  --gamma             ${GAMMA} \
  --monitor           auc \
  \
  --apply_normalization \
  --smoothing_mode    gaussian \
  --gaussian_sigma    1.0

echo "DONE (seed=${SEED}) at $(date)"
