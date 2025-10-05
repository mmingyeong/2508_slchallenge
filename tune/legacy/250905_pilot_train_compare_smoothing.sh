#!/bin/bash
#PBS -N pilot_train_smoothing
#PBS -q long
#PBS -l nodes=1:ppn=4:gpus=1
#PBS -l mem=32gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -M mmingyeong@kasi.re.kr
#PBS -o pilot_train_smoothing.pbs.out

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

# Smoothing hyper-params
GAUSS_SIGMA=0.8          # pixels
GUIDED_RADIUS=2          # pixels (window radius)
GUIDED_EPS=1e-2          # intensity^2 units

# Normalization
CLIP_Q=0.997             # set <0 to disable clipping (plain z-score)

# Data paths
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
    \
    --apply_normalization \
    --clip_q              $CLIP_Q \
    \
    --smoothing_mode      "$@"   # pieces passed from caller

  # quick peek of final metrics
  tail -n 2 "${SAVE_DIR}/train.log" || true
  if [ -f "${SAVE_DIR}/results.json" ]; then
    echo "RESULTS (${CASE_NAME}):"; cat "${SAVE_DIR}/results.json"
  fi
}

# ----------------------------
# Three smoothing cases (NO padding, normalization ON)
# ----------------------------

# 1) No smoothing
run_case "sm_none_norm1_pad0" \
  none

# 2) Gaussian smoothing
run_case "sm_gaussian_norm1_pad0" \
  gaussian --gaussian_sigma ${GAUSS_SIGMA}

# 3) Guided filter smoothing
run_case "sm_guided_norm1_pad0" \
  guided --guided_radius ${GUIDED_RADIUS} --guided_eps ${GUIDED_EPS}

echo "DONE at $(date)"
