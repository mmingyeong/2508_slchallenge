#PBS -N sl_predict_nano_val_all
#PBS -q long
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l mem=32gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -M mmingyeong@kasi.re.kr
#PBS -o sl_predict_nano_val_all.pbs.out

# ----------------------------
# User config
# ----------------------------
PROJECT_ROOT="/home/users/mmingyeong/2508_slchallence"   # ✅ 수정된 경로
PYENV="py312"


# Data
SLSIM_LENSES="/caefs/data/IllustrisTNG/slchallenge/slsim_lenses/slsim_lenses"
SLSIM_NONLENSES="/caefs/data/IllustrisTNG/slchallenge/slsim_nonlenses/slsim_nonlenses"
HSC_LENSES="/caefs/data/IllustrisTNG/slchallenge/hsc_lenses/hsc_lenses"
HSC_NONLENSES="/caefs/data/IllustrisTNG/slchallenge/hsc_nonlenses/hsc_nonlenses"

WHICH="val"

# ----------------------------
# Environment
# ----------------------------
source ~/.bashrc
conda activate "${PYENV}"

echo "HOST    : $(hostname)"
echo "DATE    : $(date)"
echo "CUDA_DEV: ${CUDA_VISIBLE_DEVICES:-all}"
nvidia-smi || true

# ----------------------------
# Collect checkpoint directories (nano)
# ----------------------------
CKPT_ROOT="${PROJECT_ROOT}/res/nano_ckpt"
if [[ ${#CKPT_DIRS[@]:-0} -eq 0 ]]; then
  mapfile -t CKPT_DIRS < <(ls -dt "${CKPT_ROOT}"/_full_ckpt_fulltrain_nano_s*_* 2>/dev/null || true)
  if [[ ${#CKPT_DIRS[@]} -eq 0 ]]; then
    mapfile -t CKPT_DIRS < <(ls -dt "${CKPT_ROOT}"/_full_ckpt_fulltrain_nano_* 2>/dev/null || true)
  fi
fi
if [[ ${#CKPT_DIRS[@]} -eq 0 ]]; then
  echo "ERROR: No checkpoint directories found under ${CKPT_ROOT}/"
  exit 1
fi

echo "Found checkpoints:"
printf ' - %s\n' "${CKPT_DIRS[@]}"

RUN_TAG="pred_$(date +%Y%m%d_%H%M%S)"
CREATED_LIST="${PROJECT_ROOT}/res/pred_val_created_${RUN_TAG}.txt"
: > "${CREATED_LIST}"

# ----------------------------
# Loop over checkpoints
# ----------------------------
for CKPT_DIR in "${CKPT_DIRS[@]}"; do
  [[ -d "${CKPT_DIR}" ]] || continue

  MODEL_PATH="${CKPT_DIR}/best.pt"
  [[ -f "${MODEL_PATH}" ]] || MODEL_PATH="${CKPT_DIR}/last.pt"
  if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "SKIP: No best.pt/last.pt in ${CKPT_DIR}"
    continue
  fi

  CKPT_TAG="$(basename "${CKPT_DIR}")"

  # auto-detect model size (nano vs atto)
  if [[ "${CKPT_TAG}" == *"nano"* ]]; then
    MODEL_SIZE="nano"
  else
    MODEL_SIZE="atto"
  fi
  echo ">>> Predict ${WHICH} | CKPT=${CKPT_TAG} | ARCH=${MODEL_SIZE}"

  OUT_DIR="${PROJECT_ROOT}/res/${CKPT_TAG}_pred_${WHICH}_${RUN_TAG}"
  CSV_PATH="${OUT_DIR}/pred_${WHICH}.csv"
  if [[ -f "${CSV_PATH}" ]]; then
    echo "SKIP: ${CSV_PATH} already exists"
    continue
  fi
  mkdir -p "${OUT_DIR}"

  # ---- Build argv as an array to avoid line-continuation issues ----
  ARGS=(
    "${PROJECT_ROOT}/src/predict.py"
    --slsim_lenses      "${SLSIM_LENSES}"
    --slsim_nonlenses   "${SLSIM_NONLENSES}"
    --hsc_lenses        "${HSC_LENSES}"
    --hsc_nonlenses     "${HSC_NONLENSES}"
    --which             "${WHICH}"
    --batch_size        128
    --num_workers       8
    --train_frac        0.70
    --val_frac          0.15
    --test_frac         0.15
    --seed              42
    --model_path        "${MODEL_PATH}"
    --model_size        "${MODEL_SIZE}"
    --drop_path         0.05
    --device            cuda
    --output_dir        "${OUT_DIR}"
    --apply_normalization
    --smoothing_mode    gaussian
    --gaussian_sigma    1.0
  )

  echo "    MODEL=${MODEL_PATH}"
  echo "    OUT  =${OUT_DIR}"

  # run
  python -u "${ARGS[@]}"

  if [[ -f "${CSV_PATH}" ]]; then
    echo "${CSV_PATH}" | tee -a "${CREATED_LIST}"
  else
    echo "WARN: pred CSV not found for ${CKPT_TAG} (expected ${CSV_PATH})"
  fi
done

echo "Created VAL CSV list -> ${CREATED_LIST}"
echo "DONE at $(date)"
