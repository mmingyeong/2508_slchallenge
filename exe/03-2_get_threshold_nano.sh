#!/bin/bash
#PBS -N sl_eval_threshold_val_multi
#PBS -q long
#PBS -l nodes=1:ppn=2
#PBS -l mem=8gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -M mmingyeong@kasi.re.kr
#PBS -o sl_eval_threshold_val_multi.pbs.out

# ----------------------------
# User config
# ----------------------------
PROJECT_ROOT="/caefs/user/mmingyeong/2508_slchallence"
PYENV="py312"

# VAL 예측 폴더들이 위치한 베이스 경로
BASE_DIR="/home/users/mmingyeong/2508_slchallence/res"

# (옵션1) 자동 수집: 아래 패턴과 일치하는 모든 폴더를 정렬하여 처리
#         s42, s101, s202, s303, s404 순서가 자연스럽게 오도록 -V 정렬 사용
mapfile -t TARGET_DIRS < <(ls -d "${BASE_DIR}"/_full_ckpt_fulltrain_nano_s*_*/ 2>/dev/null | grep "_pred_val_pred_" | sort -V)

# (옵션2) 수동 지정: 필요 시 아래 배열을 채워서 사용 (위 자동 수집 주석 처리)
# TARGET_DIRS=(
#   "/home/users/mmingyeong/2508_slchallence/res/_full_ckpt_fulltrain_nano_s42_20250925_184205_pred_val_pred_20251005_132054"
#   "/home/users/mmingyeong/2508_slchallence/res/_full_ckpt_fulltrain_nano_s101_20250925_184206_pred_val_pred_20251005_132054"
#   "/home/users/mmingyeong/2508_slchallence/res/_full_ckpt_fulltrain_nano_s202_20250927_134111_pred_val_pred_20251005_132054"
#   "/home/users/mmingyeong/2508_slchallence/res/_full_ckpt_fulltrain_nano_s303_20250927_191302_pred_val_pred_20251005_132054"
#   "/home/users/mmingyeong/2508_slchallence/res/_full_ckpt_fulltrain_nano_s404_20250929_150029_pred_val_pred_20251005_132054"
# )

# ----------------------------
# Environment
# ----------------------------
source ~/.bashrc
conda activate "${PYENV}"

echo "HOST: $(hostname)"
echo "DATE: $(date)"

if [[ ${#TARGET_DIRS[@]} -eq 0 ]]; then
  echo "ERROR: No target directories found under ${BASE_DIR} matching *_pred_val_pred_*"
  exit 1
fi

echo "===> Found VAL pred directories:"
printf ' - %s\n' "${TARGET_DIRS[@]}"

# ----------------------------
# Per-folder processing
# ----------------------------
for TARGET_DIR in "${TARGET_DIRS[@]}"; do
  TARGET_DIR="${TARGET_DIR%/}"     # strip trailing slash
  TARGET_CSV="${TARGET_DIR}/pred_val.csv"

  echo
  echo "=============================="
  echo "Processing: ${TARGET_DIR}"
  echo "=============================="

  if [[ ! -f "${TARGET_CSV}" ]]; then
    echo "WARN: pred_val.csv not found -> ${TARGET_CSV} (skip)"
    continue
  fi

  OUT_DIR="${TARGET_DIR}/eval_val_ensemble"
  mkdir -p "${OUT_DIR}"

  # ---------- Build (degenerate) ensemble CSV ----------
  ENSEMBLED_VAL_CSV="${OUT_DIR}/val_ens.csv"

  python - <<'PY' "${ENSEMBLED_VAL_CSV}" "${TARGET_CSV}"
import sys, os, pandas as pd
out_csv = sys.argv[1]
csv_path = sys.argv[2]

if not os.path.isfile(csv_path):
    raise SystemExit(f"CSV not found: {csv_path}")

df = pd.read_csv(csv_path)
need = {"path","label","prob"}
missing = need - set(df.columns)
if missing:
    raise SystemExit(f"{csv_path} missing columns: {missing}")

# 정수 라벨 보장
df["label"] = pd.to_numeric(df["label"], errors="raise").astype(float).round().astype(int)

# 단일 CSV라도 동일 인터페이스 유지: prob를 그대로 사용
ens = df[["path","label","prob"]].copy()
ens.to_csv(out_csv, index=False)
print(f"Ensembled (single or multi) VAL CSV -> {out_csv} (N={len(ens)})")
PY

  # ---------- Evaluate & optimize threshold ----------
  python -u "${PROJECT_ROOT}/src/evaluate.py" \
    --tag "val_ens" \
    --from_csv --csv "${ENSEMBLED_VAL_CSV}" \
    --optimize_threshold \
    --out_dir "${OUT_DIR}"

  METRICS_JSON="${OUT_DIR}/metrics_val_ens.json"

  # ---------- Extract and save threshold t* ----------
  VAL_TH=$(
  python - <<PY
import json, sys
with open(sys.argv[1], "r") as f:
    m = json.load(f)
th = m.get("global", {}).get("threshold", None)
if th is None:
    raise SystemExit("threshold not found")
print(th)
PY
  "${METRICS_JSON}"
  )

  echo "Optimal threshold t* (VAL) = ${VAL_TH}"
  echo "${VAL_TH}" > "${OUT_DIR}/threshold.txt"

  echo "Saved:"
  echo " - ${ENSEMBLED_VAL_CSV}"
  echo " - ${METRICS_JSON}"
  echo " - ${OUT_DIR}/threshold.txt"
done

echo
echo "ALL DONE at $(date)"

# ----------------------------------------------------
# (선택 사항) 모든 폴더의 VAL CSV를 합쳐 '교차-시드 앙상블'을 원하면,
# 아래 블록을 주석 해제하여 추가 실행하세요.
# ----------------------------------------------------
# COMBO_OUT="${BASE_DIR}/eval_val_ensemble_all"
# mkdir -p "${COMBO_OUT}"
# COMBO_CSV="${COMBO_OUT}/val_ens_all.csv"
#
# # 모든 pred_val.csv 모으기
# mapfile -t ALL_CSVS < <(for d in "${TARGET_DIRS[@]}"; do
#   d="${d%/}"
#   if [[ -f "${d}/pred_val.csv" ]]; then echo "${d}/pred_val.csv"; fi
# done)
#
# if [[ ${#ALL_CSVS[@]} -gt 0 ]]; then
#   python - <<'PY' "${COMBO_CSV}" "${#ALL_CSVS[@]}" "${ALL_CSVS[@]}"
# import sys, os, pandas as pd
# out_csv = sys.argv[1]
# n = int(sys.argv[2]); csvs = sys.argv[3:3+n]
# assert n>0, "No CSVs"
# dfs=[]
# for p in csvs:
#     if not os.path.isfile(p): continue
#     df = pd.read_csv(p)
#     need = {"path","label","prob"}
#     if not need.issubset(df.columns):
#         raise SystemExit(f"{p} missing {need - set(df.columns)}")
#     df = df[["path","label","prob"]].copy()
#     df["label"] = pd.to_numeric(df["label"], errors="raise").round().astype(int)
#     seed = os.path.basename(os.path.dirname(p))
#     df.rename(columns={"prob": seed}, inplace=True)
#     dfs.append(df)
# assert dfs, "No valid CSVs"
# ens = dfs[0]
# for d in dfs[1:]:
#     ens = ens.merge(d, on=["path","label"], how="inner")
# seed_cols = [c for c in ens.columns if c not in ("path","label")]
# ens["prob"] = ens[seed_cols].mean(axis=1)
# ens[["path","label","prob"]].to_csv(out_csv, index=False)
# print(f"Cross-seed Ensembled VAL -> {out_csv} (N={len(ens)}, seeds={len(seed_cols)})")
# PY
#
#   python -u "${PROJECT_ROOT}/src/evaluate.py" \
#     --tag "val_ens_all" \
#     --from_csv --csv "${COMBO_CSV}" \
#     --optimize_threshold \
#     --out_dir "${COMBO_OUT}"
# fi
