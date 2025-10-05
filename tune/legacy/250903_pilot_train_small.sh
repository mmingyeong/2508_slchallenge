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
# User config (edit here)
# ----------------------------
PROJECT_ROOT="/caefs/user/mmingyeong/2508_slchallence"
PYENV="py312"                # 필요시 conda env 이름
MODEL_SIZE="atto"            # atto | nano | tiny
EPOCHS=12                    # 10~15 권장
PATIENCE=5
TAKE_TRAIN_FRAC=0.01         # 0.001~0.01 권장
TAKE_VAL_FRAC=0.01           # 빠른 검증 위해 소량
TAKE_TEST_FRAC=0.01          # 빠른 테스트 위해 소량
BATCH=128
NUM_WORKERS=8
SEED=42
SAVE_DIR="${PROJECT_ROOT}/_pilot_ckpt_$(date +%Y%m%d_%H%M%S)"

# 데이터 경로
SLSIM_LENSES="/caefs/data/IllustrisTNG/slchallenge/slsim_lenses/slsim_lenses"
SLSIM_NONLENSES="/caefs/data/IllustrisTNG/slchallenge/slsim_nonlenses/slsim_nonlenses"
HSC_LENSES="/caefs/data/IllustrisTNG/slchallenge/hsc_lenses/hsc_lenses"
HSC_NONLENSES="/caefs/data/IllustrisTNG/slchallenge/hsc_nonlenses/hsc_nonlenses"

# ----------------------------
# Environment
# ----------------------------
source ~/.bashrc
conda activate py312

echo "HOST    : $(hostname)"
echo "PWD     : $(pwd)"
echo "DATE    : $(date)"
echo "CUDA_DEV: ${CUDA_VISIBLE_DEVICES:-all}"
nvidia-smi || true

# ----------------------------
# Run (no augmentation, BCEWithLogitsLoss 기본)
# ----------------------------
mkdir -p "$SAVE_DIR"

python -u "${PROJECT_ROOT}/src/train.py" \
  --slsim_lenses      "$SLSIM_LENSES" \
  --slsim_nonlenses   "$SLSIM_NONLENSES" \
  --hsc_lenses        "$HSC_LENSES" \
  --hsc_nonlenses     "$HSC_NONLENSES" \
  --batch_size        $BATCH \
  --num_workers       $NUM_WORKERS \
  --no_augment \
  --take_train_frac   $TAKE_TRAIN_FRAC \
  --take_val_fraction $TAKE_VAL_FRAC \
  --take_test_fraction $TAKE_TEST_FRAC \
  --train_frac        0.70 \
  --val_frac          0.15 \
  --test_frac         0.15 \
  --model_size        "$MODEL_SIZE" \
  --drop_path         0.0 \
  --lr                1e-3 \
  --weight_decay      1e-4 \
  --epochs            $EPOCHS \
  --patience          $PATIENCE \
  --min_delta         0.0 \
  --seed              $SEED \
  --device            cuda \
  --save_dir          "$SAVE_DIR"

echo "DONE at $(date)"
