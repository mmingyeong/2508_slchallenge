#!/bin/bash
#PBS -N sl_eval_threshold_val_ens
#PBS -q long
#PBS -l nodes=1:ppn=2
#PBS -l mem=8gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -M mmingyeong@kasi.re.kr
#PBS -o sl_eval_threshold_val_ens.pbs.out


# ----------------------------
# User config
# ----------------------------
PROJECT_ROOT="/caefs/user/mmingyeong/2508_slchallence"
PYENV="py312"

# (선택) 수동으로 사용할 VAL CSV 목록을 지정하고 싶으면 여기 배열에 채우세요.
# CSV_LIST=(
#   "/.../res/_full_ckpt_s42_.../pred_val/pred_val.csv"
#   "/.../res/_full_ckpt_s101_.../pred_val/pred_val.csv"
# )

# ----------------------------
# Environment
# ----------------------------
source ~/.bashrc
conda activate "${PYENV}"

echo "HOST: $(hostname)"
echo "DATE: $(date)"

# ----------------------------
# Collect VAL CSVs (auto)
# ----------------------------
if [[ ${#CSV_LIST[@]:-0} -eq 0 ]]; then
  mapfile -t CSV_LIST < <( \
    ls -t "${PROJECT_ROOT}"/res/*_pred_val_*/pred_val.csv 2>/dev/null; \
    ls -t "${PROJECT_ROOT}"/res/_full_ckpt_*/pred_val/pred_val.csv 2>/dev/null \
  )
fi

if [[ ${#CSV_LIST[@]} -eq 0 ]]; then
  echo "ERROR: pred_val.csv 파일을 찾지 못했습니다. 먼저 각 시드에 대해 predict.py --which val을 실행하세요."
  exit 1
fi

echo "Found VAL CSVs:"
printf ' - %s\n' "${CSV_LIST[@]}"

OUT_DIR="${PROJECT_ROOT}/res/eval_val_ensemble"
mkdir -p "${OUT_DIR}"

# ----------------------------
# Build ensemble VAL CSV (path별 확률 평균)
# ----------------------------
ENSEMBLED_VAL_CSV="${OUT_DIR}/val_ens.csv"

python - <<'PY' "${ENSEMBLED_VAL_CSV}" "${PROJECT_ROOT}" "${#CSV_LIST[@]}" "${CSV_LIST[@]}"
import sys, os, pandas as pd, numpy as np
out_csv = sys.argv[1]
project_root = sys.argv[2]
n = int(sys.argv[3])
csvs = sys.argv[4:4+n]
if n == 0:
    raise SystemExit("No CSVs provided")

dfs = []
for p in csvs:
    if not os.path.isfile(p):
        continue
    df = pd.read_csv(p)
    need = {"path","label","prob"}
    if not need.issubset(df.columns):
        raise SystemExit(f"{p} missing columns {need - set(df.columns)}")
    df = df[["path","label","prob"]].copy()

    # 🔧 label을 확실히 정수로 고정
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(float).round().astype(int)

    seed_name = os.path.basename(os.path.dirname(p))
    df.rename(columns={"prob": seed_name}, inplace=True)
    dfs.append(df)


if not dfs:
    raise SystemExit("No valid CSVs to ensemble")

# inner join으로 공통 샘플 교집합 사용 (검증 split 고정이면 동일)
ens = dfs[0]
for d in dfs[1:]:
    ens = ens.merge(d, on=["path","label"], how="inner")

# 🔧 혹시 모를 dtype 흔들림 대비
ens["label"] = pd.to_numeric(ens["label"], errors="raise").astype(float).round().astype(int)

seed_cols = [c for c in ens.columns if c not in ("path","label")]
ens["prob"] = ens[seed_cols].mean(axis=1)
ens[["path","label","prob"]].to_csv(out_csv, index=False)
print(f"Ensembled VAL CSV -> {out_csv} (N={len(ens)}, seeds={len(seed_cols)})")
PY

# ----------------------------
# Evaluate (Youden’s J로 임계값 최적화)
# ----------------------------
python -u "${PROJECT_ROOT}/src/evaluate.py" \
  --tag "val_ens" \
  --from_csv --csv "${ENSEMBLED_VAL_CSV}" \
  --optimize_threshold \
  --out_dir "${OUT_DIR}"

METRICS_JSON="${OUT_DIR}/metrics_val_ens.json"

# ----------------------------
# Extract and save threshold t*
# ----------------------------
VAL_TH=$(
python - <<PY
import json, sys
with open(sys.argv[1],"r") as f:
    m = json.load(f)
th = m.get("global", {}).get("threshold", None)
if th is None:
    raise SystemExit("threshold not found")
print(th)
PY
"${METRICS_JSON}"
)
echo "Optimal threshold t* (ensemble) = ${VAL_TH}"
echo "${VAL_TH}" > "${OUT_DIR}/threshold.txt"

echo "Saved:"
echo " - ${ENSEMBLED_VAL_CSV}"
echo " - ${METRICS_JSON}"
echo " - ${OUT_DIR}/threshold.txt"

echo "DONE at $(date)"
