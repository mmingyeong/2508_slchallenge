#!/bin/bash
#PBS -N sl_predict_atto_test_all
#PBS -q long
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l mem=32gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -M mmingyeong@kasi.re.kr
#PBS -o sl_predict_atto_test_all.pbs.out


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

# 체크포인트 자동 수집
if [[ ${#CKPT_DIRS[@]:-0} -eq 0 ]]; then
  mapfile -t CKPT_DIRS < <(ls -dt "${PROJECT_ROOT}"/res/_full_ckpt_s*_* 2>/dev/null || true)
  if [[ ${#CKPT_DIRS[@]} -eq 0 ]]; then
    mapfile -t CKPT_DIRS < <(ls -dt "${PROJECT_ROOT}"/res/_full_ckpt_* 2>/dev/null || true)
  fi
fi
[[ ${#CKPT_DIRS[@]} -gt 0 ]] || { echo "ERROR: no checkpoints"; exit 1; }

RUN_TAG="pred_$(date +%Y%m%d_%H%M%S)"

for CKPT_DIR in "${CKPT_DIRS[@]}"; do
  [[ -d "${CKPT_DIR}" ]] || continue
  MODEL_PATH="${CKPT_DIR}/best.pt"; [[ -f "$MODEL_PATH" ]] || MODEL_PATH="${CKPT_DIR}/last.pt"
  [[ -f "$MODEL_PATH" ]] || { echo "SKIP: no ckpt in ${CKPT_DIR}"; continue; }

  CKPT_TAG="$(basename "${CKPT_DIR}")"
  OUT_DIR="${PROJECT_ROOT}/res/${CKPT_TAG}_pred_${WHICH}_${RUN_TAG}"
  CSV_PATH="${OUT_DIR}/pred_${WHICH}.csv"
  [[ -f "${CSV_PATH}" ]] && { echo "SKIP: ${CSV_PATH} exists"; continue; }

  mkdir -p "${OUT_DIR}"
  echo ">>> Predict ${WHICH} | ${CKPT_TAG}"

  python -u "${PROJECT_ROOT}/src/predict.py" \
    --slsim_lenses "${SLSIM_LENSES}" \
    --slsim_nonlenses "${SLSIM_NONLENSES}" \
    --hsc_lenses   "${HSC_LENSES}" \
    --hsc_nonlenses "${HSC_NONLENSES}" \
    --which "${WHICH}" \
    --batch_size 128 --num_workers 8 \
    --train_frac 0.70 --val_frac 0.15 --test_frac 0.15 \
    --seed 42 \
    --model_path "${MODEL_PATH}" \
    --model_size "atto" \
    --drop_path 0.05 \
    --device cuda \
    --output_dir "${OUT_DIR}" \
    --apply_normalization \
    --smoothing_mode gaussian \
    --gaussian_sigma 1.0
done

echo "DONE at $(date)"
