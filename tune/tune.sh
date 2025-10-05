#!/bin/bash
#PBS -N sl_microhpo_inmem
#PBS -q long
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l mem=48gb
#PBS -l walltime=36:00:00
#PBS -j oe
#PBS -M mmingyeong@kasi.re.kr
#PBS -o sl_microhpo_inmem.pbs.out

# --- 사용자 설정 ---
PROJECT_ROOT="/caefs/user/mmingyeong/2508_slchallence"
SRC_DIR="${PROJECT_ROOT}/src"
PYENV="py312"

# 데이터 경로
SLSIM_LENSES="/caefs/data/IllustrisTNG/slchallenge/slsim_lenses/slsim_lenses"
SLSIM_NONLENSES="/caefs/data/IllustrisTNG/slchallenge/slsim_nonlenses/slsim_nonlenses"
HSC_LENSES="/caefs/data/IllustrisTNG/slchallenge/hsc_lenses/hsc_lenses"
HSC_NONLENSES="/caefs/data/IllustrisTNG/slchallenge/hsc_nonlenses/hsc_nonlenses"

# 출력 경로
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${PROJECT_ROOT}/res/micro_hpo/${RUN_TAG}"

# 튜닝 trial 수(단일 워커)
N_TRIALS=12

# 고정 학습 설정 (마이크로-HPO 기준)
FIXED_ARCH="atto"
FIXED_BATCH=128
FIXED_LR=3.5e-4

# 학습 길이/조기종료 (탐색 안정성용)
EPOCHS=60
PATIENCE=15
DROP_PATH=0.05

# 데이터 샘플링 (탐색 가속)
TAKE_TRAIN_FRAC=0.10
TAKE_VAL_FRAC=0.20
NUM_WORKERS=8
DEVICE="cuda"
SEED=42

# 탐색 공간: weight decay + 4 스케줄러
WD_MIN=3e-5
WD_MAX=3e-4
SCHEDULERS=("cosine" "step" "exp" "plateau")

# Cosine warmup
WARMUP_PCT_MIN=0.03
WARMUP_PCT_MAX=0.10
MIN_LR=1e-6   # cosine/plateau에서 LR 하한

# StepLR
STEP_MIN=8            # epoch 단위
STEP_MAX=30
STEP_GAMMA_MIN=0.3
STEP_GAMMA_MAX=0.8

# ExponentialLR
EXP_GAMMA_MIN=0.90
EXP_GAMMA_MAX=0.999

# ReduceLROnPlateau
PL_FACTOR_MIN=0.2
PL_FACTOR_MAX=0.7
PL_PATIENCE_MIN=2
PL_PATIENCE_MAX=6
PL_COOLDOWN_MIN=0
PL_COOLDOWN_MAX=4
PL_MONITOR="loss"     # "loss" 또는 "auc" 중 택1 (train.py와 일치)

# --- 환경 ---
source ~/.bashrc
conda activate "${PYENV}"
mkdir -p "${OUT_ROOT}"
echo "HOST: $(hostname)"; echo "DATE: $(date)"; nvidia-smi || true

# 재현을 위한 선택적 설정(원하면 주석 해제)
# export CUBLAS_WORKSPACE_CONFIG=:4096:8
# export PYTHONHASHSEED=${SEED}

# --- 실행 ---
python -u "${PROJECT_ROOT}/src/tune.py" \
  --src_dir "${SRC_DIR}" \
  --slsim_lenses "${SLSIM_LENSES}" \
  --slsim_nonlenses "${SLSIM_NONLENSES}" \
  --hsc_lenses "${HSC_LENSES}" \
  --hsc_nonlenses "${HSC_NONLENSES}" \
  --out_root "${OUT_ROOT}" \
  \
  --epochs ${EPOCHS} --patience ${PATIENCE} --drop_path ${DROP_PATH} \
  --take_train_frac ${TAKE_TRAIN_FRAC} --take_val_fraction ${TAKE_VAL_FRAC} \
  --workers ${NUM_WORKERS} --device ${DEVICE} --seed ${SEED} \
  \
  --fixed_arch "${FIXED_ARCH}" --fixed_batch ${FIXED_BATCH} --fixed_lr ${FIXED_LR} \
  --wd_min ${WD_MIN} --wd_max ${WD_MAX} \
  --scheduler_choices "${SCHEDULERS[@]}" \
  \
  --warmup_pct_min ${WARMUP_PCT_MIN} --warmup_pct_max ${WARMUP_PCT_MAX} \
  --min_lr ${MIN_LR} \
  \
  --step_min ${STEP_MIN} --step_max ${STEP_MAX} \
  --step_gamma_min ${STEP_GAMMA_MIN} --step_gamma_max ${STEP_GAMMA_MAX} \
  \
  --exp_gamma_min ${EXP_GAMMA_MIN} --exp_gamma_max ${EXP_GAMMA_MAX} \
  \
  --plateau_factor_min ${PL_FACTOR_MIN} --plateau_factor_max ${PL_FACTOR_MAX} \
  --plateau_patience_min ${PL_PATIENCE_MIN} --plateau_patience_max ${PL_PATIENCE_MAX} \
  --plateau_cooldown_min ${PL_COOLDOWN_MIN} --plateau_cooldown_max ${PL_COOLDOWN_MAX} \
  --plateau_monitor ${PL_MONITOR} \
  \
  --apply_normalization --smoothing_mode gaussian --gaussian_sigma 1.0 \
  --n_trials ${N_TRIALS}

echo "DONE at $(date)"
