#!/bin/bash
#PBS -N pilot_train_small
#PBS -q long
#PBS -l nodes=1:ppn=4:gpus=1
#PBS -l mem=32gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -M mmingyeong@kasi.re.kr
#PBS -o pilot_train_small.pbs.out

# ----------------------------
# User config
# ----------------------------
PROJECT_ROOT="/caefs/user/mmingyeong/2508_slchallence"
PYENV="py312"
MODEL_SIZE="atto"
EPOCHS=12
PATIENCE=5
TAKE_TRAIN_FRAC=0.01
TAKE_VAL_FRAC=0.01
TAKE_TEST_FRAC=0.01
BATCH=128
NUM_WORKERS=8
SEED=42
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
BASE_SAVE_DIR="${PROJECT_ROOT}/_pilot_ckpt_${RUN_TAG}"

SLSIM_LENSES="/caefs/data/IllustrisTNG/slchallenge/slsim_lenses/slsim_lenses"
SLSIM_NONLENSES="/caefs/data/IllustrisTNG/slchallenge/slsim_nonlenses/slsim_nonlenses"
HSC_LENSES="/caefs/data/IllustrisTNG/slchallenge/hsc_lenses/hsc_lenses"
HSC_NONLENSES="/caefs/data/IllustrisTNG/slchallenge/hsc_nonlenses/hsc_nonlenses"

# ----------------------------
# Env
# ----------------------------
source ~/.bashrc
conda activate "${PYENV}"
mkdir -p "${BASE_SAVE_DIR}"

echo "HOST    : $(hostname)"
echo "PWD     : $(pwd)"
echo "DATE    : $(date)"
echo "CUDA_DEV: ${CUDA_VISIBLE_DEVICES:-all}"
nvidia-smi || true

# ----------------------------
# Helper to run one case
# ----------------------------
run_case () {
  local CASE_NAME="$1"
  shift
  local SAVE_DIR="${BASE_SAVE_DIR}/${CASE_NAME}"
  mkdir -p "${SAVE_DIR}"

  echo "===== Running case: ${CASE_NAME} ====="
  python -u "${PROJECT_ROOT}/src/train.py" \
    --slsim_lenses        "$SLSIM_LENSES" \
    --slsim_nonlenses     "$SLSIM_NONLENSES" \
    --hsc_lenses          "$HSC_LENSES" \
    --hsc_nonlenses       "$HSC_NONLENSES" \
    --batch_size          $BATCH \
    --num_workers         $NUM_WORKERS \
    --no_augment \
    --take_train_frac     $TAKE_TRAIN_FRAC \
    --take_val_fraction   $TAKE_VAL_FRAC \
    --take_test_fraction  $TAKE_TEST_FRAC \
    --train_frac          0.70 \
    --val_frac            0.15 \
    --test_frac           0.15 \
    --model_size          "$MODEL_SIZE" \
    --drop_path           0.0 \
    --lr                  1e-3 \
    --weight_decay        1e-4 \
    --epochs              $EPOCHS \
    --patience            $PATIENCE \
    --min_delta           0.0 \
    --seed                $SEED \
    --device              cuda \
    --save_dir            "$SAVE_DIR" \
    "$@"

  # quick peek of final metrics (also saved in results.json)
  tail -n 2 "${SAVE_DIR}/train.log" || true
  if [ -f "${SAVE_DIR}/results.json" ]; then
    echo "RESULTS (${CASE_NAME}):"; cat "${SAVE_DIR}/results.json"
  fi
}

# ----------------------------
# Four cases
# ----------------------------
# 1) raw: no padding, no normalization
run_case "raw_pad0_norm0" \
  # no extra flags

# 2) padding only
run_case "pad1_norm0" \
  --apply_padding \
  --out_size_when_padded 64

# 3) normalization only (plain z-score with clipping; set --clip_q -1 to disable clipping)
run_case "pad0_norm1" \
  --apply_normalization \
  --clip_q 0.997 \
  --out_size_when_padded 64   # ignored because no --apply_padding

# 4) padding + normalization
run_case "pad1_norm1" \
  --apply_padding \
  --out_size_when_padded 64 \
  --apply_normalization \
  --clip_q 0.997

echo "DONE at $(date)"
