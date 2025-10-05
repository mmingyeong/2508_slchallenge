#!/bin/bash
#PBS -N sl_predict_test_all
#PBS -q long
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l mem=32gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -M mmingyeong@kasi.re.kr
#PBS -o sl_predict_test_all.pbs.out

PROJECT_ROOT="/caefs/user/mmingyeong/2508_slchallence"
PYENV="py312"

SLSIM_LENSES="/caefs/data/IllustrisTNG/slchallenge/slsim_lenses/slsim_lenses"
SLSIM_NONLENSES="/caefs/data/IllustrisTNG/slchallenge/slsim_nonlenses/slsim_nonlenses"
HSC_LENSES="/caefs/data/IllustrisTNG/slchallenge/hsc_lenses/hsc_lenses"
HSC_NONLENSES="/caefs/data/IllustrisTNG/slchallenge/hsc_nonlenses/hsc_nonlenses"

WHICH="test"

source ~/.bashrc
conda activate "${PYENV}"

echo "HOST: $(hostname)"; echo "DATE: $(date)"; nvidia-smi || true

# ----------------------------
# Collect checkpoints from nano_ckpt/
# ----------------------------
CKPT_ROOT="${PROJECT_ROOT}/res/nano_ckpt"
if [[ ${#CKPT_DIRS[@]:-0} -eq 0 ]]; then
  mapfile -t CKPT_DIRS < <(ls -dt "${CKPT_ROOT}"/_full_ckpt_fulltrain_*_s*_* 2>/dev/null || true)
fi
[[ ${#CKPT_DIRS[@]} -gt 0 ]] || { echo "ERROR: no checkpoints under ${CKPT_ROOT}"; exit 1; }

RUN_TAG="pred_$(date +%Y%m%d_%H%M%S)"

for CKPT_DIR in "${CKPT_DIRS[@]}"; do
  [[ -d "${CKPT_DIR}" ]] || continue
  MODEL_PATH="${CKPT_DIR}/best.pt"; [[ -f "$MODEL_PATH" ]] || MODEL_PATH="${CKPT_DIR}/last.pt"
  [[ -f "$MODEL_PATH" ]] || { echo "SKIP: no best.pt/last.pt in ${CKPT_DIR}"; continue; }

  CKPT_TAG="$(basename "${CKPT_DIR}")"

  # ---- ARCH auto-detect from tag ----
  if   [[ "${CKPT_TAG}" == *"_nano_"* ]]; then MODEL_SIZE="nano"
  elif [[ "${CKPT_TAG}" == *"_atto_"* ]]; then MODEL_SIZE="atto"
  elif [[ "${CKPT_TAG}" == *"_tiny_"* ]]; then MODEL_SIZE="tiny"
  else
    # (안전) 기본값: nano 로 가정
    MODEL_SIZE="nano"
    echo "WARN: cannot infer model_size from tag '${CKPT_TAG}', defaulting to ${MODEL_SIZE}"
  fi

  OUT_DIR="${CKPT_ROOT}/${CKPT_TAG}_pred_${WHICH}_${RUN_TAG}"
  CSV_PATH="${OUT_DIR}/pred_${WHICH}.csv"
  [[ -f "${CSV_PATH}" ]] && { echo "SKIP: ${CSV_PATH} exists"; continue; }
  mkdir -p "${OUT_DIR}"

  echo ">>> Predict ${WHICH} | ${CKPT_TAG} | ARCH=${MODEL_SIZE}"
  echo "    MODEL=${MODEL_PATH}"
  echo "    OUT  =${OUT_DIR}"

  # 안전하게 배열로 인자 전달 (라인 연속 이슈 방지)
  python -u "${PROJECT_ROOT}/src/predict.py" \
    --slsim_lenses   "${SLSIM_LENSES}" \
    --slsim_nonlenses "${SLSIM_NONLENSES}" \
    --hsc_lenses     "${HSC_LENSES}" \
    --hsc_nonlenses  "${HSC_NONLENSES}" \
    --which "${WHICH}" \
    --batch_size 128 \
    --num_workers 8 \
    --train_frac 0.70 \
    --val_frac   0.15 \
    --test_frac  0.15 \
    --seed 42 \
    --model_path "${MODEL_PATH}" \
    --model_size "${MODEL_SIZE}" \
    --drop_path 0.05 \
    --device cuda \
    --output_dir "${OUT_DIR}" \
    --apply_normalization \
    --smoothing_mode gaussian \
    --gaussian_sigma 1.0

  if [[ -f "${CSV_PATH}" ]]; then
    echo "OK: ${CSV_PATH}"
  else
    echo "WARN: pred CSV not found for ${CKPT_TAG} (expected ${CSV_PATH})"
  fi
done

echo "DONE at $(date)"
