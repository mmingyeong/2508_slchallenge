#!/bin/bash
#PBS -N sl_predict_atto_val_all
#PBS -q long
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l mem=32gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -M mmingyeong@kasi.re.kr
#PBS -o sl_predict_atto_val_all.pbs.out


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

# Predict target split
WHICH="val"

# (선택) 수동 지정: 특정 체크포인트 디렉터리 목록을 직접 지정하려면 아래 배열 사용
# CKPT_DIRS=( "/path/to/_full_ckpt_s42_..." "/path/to/_full_ckpt_s101_..." )

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
# Collect checkpoint directories (auto)
# ----------------------------
if [[ ${#CKPT_DIRS[@]:-0} -eq 0 ]]; then
  # 시드 표기가 있는 디렉터리 우선 수집
  mapfile -t CKPT_DIRS < <(ls -dt "${PROJECT_ROOT}"/res/_full_ckpt_s*_* 2>/dev/null || true)
  # 없으면 일반 패턴으로 fallback
  if [[ ${#CKPT_DIRS[@]} -eq 0 ]]; then
    mapfile -t CKPT_DIRS < <(ls -dt "${PROJECT_ROOT}"/res/_full_ckpt_* 2>/dev/null || true)
  fi
fi

if [[ ${#CKPT_DIRS[@]} -eq 0 ]]; then
  echo "ERROR: No checkpoint directories found under ${PROJECT_ROOT}/res/"
  exit 1
fi

echo "Found checkpoints:"
printf ' - %s\n' "${CKPT_DIRS[@]}"

RUN_TAG="pred_$(date +%Y%m%d_%H%M%S)"
CREATED_LIST="${PROJECT_ROOT}/res/pred_val_created_${RUN_TAG}.txt"
: > "${CREATED_LIST}"

# ----------------------------
# Loop over checkpoints (sequential)
# ----------------------------
for CKPT_DIR in "${CKPT_DIRS[@]}"; do
  [[ -d "${CKPT_DIR}" ]] || continue

  MODEL_PATH="${CKPT_DIR}/best.pt"
  if [[ ! -f "${MODEL_PATH}" ]]; then
    MODEL_PATH="${CKPT_DIR}/last.pt"
  fi
  if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "SKIP: No best.pt/last.pt in ${CKPT_DIR}"
    continue
  fi

  CKPT_TAG="$(basename "${CKPT_DIR}")"
  OUT_DIR="${PROJECT_ROOT}/res/${CKPT_TAG}_pred_${WHICH}_${RUN_TAG}"
  CSV_PATH="${OUT_DIR}/pred_${WHICH}.csv"

  # 이미 생성된 경우 스킵
  if [[ -f "${CSV_PATH}" ]]; then
    echo "SKIP: ${CSV_PATH} already exists"
    continue
  fi

  mkdir -p "${OUT_DIR}"
  echo ">>> Predict ${WHICH} | CKPT=${CKPT_TAG}"
  echo "    MODEL=${MODEL_PATH}"
  echo "    OUT  =${OUT_DIR}"

  # Predict (증강 OFF, 학습과 동일 전처리)
  python -u "${PROJECT_ROOT}/src/predict.py" \
    --slsim_lenses      "${SLSIM_LENSES}" \
    --slsim_nonlenses   "${SLSIM_NONLENSES}" \
    --hsc_lenses        "${HSC_LENSES}" \
    --hsc_nonlenses     "${HSC_NONLENSES}" \
    --which             "${WHICH}" \
    \
    --batch_size        128 \
    --num_workers       8 \
    --train_frac        0.70 \
    --val_frac          0.15 \
    --test_frac         0.15 \
    --seed              42 \
    \
    --model_path        "${MODEL_PATH}" \
    --model_size        "atto" \
    --drop_path         0.05 \
    --device            cuda \
    --output_dir        "${OUT_DIR}" \
    \
    --apply_normalization \
    --smoothing_mode    gaussian \
    --gaussian_sigma    1.0

  if [[ -f "${CSV_PATH}" ]]; then
    echo "${CSV_PATH}" | tee -a "${CREATED_LIST}"
  else
    echo "WARN: pred CSV not found for ${CKPT_TAG} (expected ${CSV_PATH})"
  fi
done

echo "Created VAL CSV list -> ${CREATED_LIST}"
echo "DONE at $(date)"
