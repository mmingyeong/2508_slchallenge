#!/bin/bash
#PBS -N sl_eval_labeled_test_with_tstar_hybrid
#PBS -q long
#PBS -l nodes=1:ppn=2
#PBS -l mem=8gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -M mmingyeong@kasi.re.kr
#PBS -o sl_eval_labeled_test_with_tstar_hybrid.pbs.out

# ----------------------------
# User config
# ----------------------------
PYENV="py312"

# ATTO/NANO 결과 루트 (스크린샷 기준)
ATTO_ROOT="/home/users/mmingyeong/2508_slchallence/res"
NANO_ROOT="/home/users/mmingyeong/2508_slchallence/res/nano_ckpt"

# evaluate.py 경로 (필요하면 수정)
EVAL_PY="/caefs/user/mmingyeong/2508_slchallence/src/evaluate.py"

RUN_TAG="test_eval_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ATTO_ROOT}/ensemble_hybrid_test_${RUN_TAG}"
mkdir -p "${OUT_DIR}"

# ----------------------------
# Environment
# ----------------------------
source ~/.bashrc
conda activate "${PYENV}"

echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "ATTO_ROOT = ${ATTO_ROOT}"
echo "NANO_ROOT = ${NANO_ROOT}"
echo "EVAL_PY   = ${EVAL_PY}"

# ----------------------------
# Load t* (prefer hybrid, else latest from atto/nano; else json; else 0.5)
# ----------------------------
if [[ -z "${THRESHOLD_FILE:-}" ]]; then
  # 1) 하이브리드 VAL
  if [[ -f "${ATTO_ROOT}/ensemble_hybrid_val/threshold.txt" ]]; then
    THRESHOLD_FILE="${ATTO_ROOT}/ensemble_hybrid_val/threshold.txt"
  else
    # 2) atto/nano의 eval_val_ensemble/threshold.txt 중 최신
    THRESHOLD_FILE=$(ls -t \
      "${ATTO_ROOT}"/**/eval_val_ensemble/threshold.txt \
      "${NANO_ROOT}"/**/eval_val_ensemble/threshold.txt \
      2>/dev/null | head -n1 || true)
  fi
fi

if [[ -n "${THRESHOLD_FILE:-}" && -f "${THRESHOLD_FILE}" ]]; then
  TSTAR=$(cat "${THRESHOLD_FILE}")
  echo "Using t* = ${TSTAR}  (from ${THRESHOLD_FILE})"
else
  echo "threshold.txt not found; trying metrics_val_ens.json ..."
  # 3) metrics_val_ens.json에서 threshold 추출
  MET_JSON=$(ls -t \
    "${ATTO_ROOT}"/**/eval_val_ensemble/metrics_val_ens.json \
    "${NANO_ROOT}"/**/eval_val_ensemble/metrics_val_ens.json \
    2>/dev/null | head -n1 || true)

  if [[ -n "${MET_JSON}" && -f "${MET_JSON}" ]]; then
    TSTAR=$(
      python - <<'PY' "${MET_JSON}"
import sys, json
p = sys.argv[1]
with open(p, "r") as f:
    m = json.load(f)
thr = None
try:
    thr = float(m.get("global", {}).get("threshold"))
except Exception:
    thr = None
if thr is None:
    # scan any 0..1 numeric
    def scan(d):
        if isinstance(d, dict):
            for v in d.values():
                r = scan(v)
                if r is not None: return r
        elif isinstance(d, (int,float)) and 0.0 <= d <= 1.0:
            return float(d)
    thr = scan(m)
if thr is None:
    thr = 0.5
print(thr)
PY
    )
    echo "Using t* = ${TSTAR}  (from ${MET_JSON})"
  else
    TSTAR="0.5"
    echo "WARN: No threshold source found; fallback t* = ${TSTAR}"
  fi
fi

# ----------------------------
# Collect TEST CSVs from both trees (dedup)
# ----------------------------
mapfile -t _FOUND < <(
  # ATTO: 신규 *_pred_test_pred_*/pred_test.csv
  ls -t "${ATTO_ROOT}"/_full_ckpt_fulltrain_s*_pred_test_pred_*/*/pred_test.csv 2>/dev/null || true
  ls -t "${ATTO_ROOT}"/_full_ckpt_fulltrain_s*_pred_test_pred_*/pred_test.csv        2>/dev/null || true
  # ATTO: 구형 pred_test/pred_test.csv
  ls -t "${ATTO_ROOT}"/_full_ckpt_fulltrain_s*/pred_test/pred_test.csv               2>/dev/null || true

  # NANO: 신규 *_pred_test_pred_*/pred_test.csv
  ls -t "${NANO_ROOT}"/_full_ckpt_fulltrain_nano_s*_pred_test_pred_*/*/pred_test.csv 2>/dev/null || true
  ls -t "${NANO_ROOT}"/_full_ckpt_fulltrain_nano_s*_pred_test_pred_*/pred_test.csv   2>/dev/null || true
  # NANO: 구형 pred_test/pred_test.csv
  ls -t "${NANO_ROOT}"/_full_ckpt_fulltrain_nano_s*/pred_test/pred_test.csv          2>/dev/null || true
)

declare -A SEEN=()
TEST_CSV_LIST=()
for p in "${_FOUND[@]:-}"; do
  [[ -f "$p" ]] || continue
  if [[ -z "${SEEN[$p]:-}" ]]; then
    SEEN[$p]=1
    TEST_CSV_LIST+=("$p")
  fi
done

if [[ ${#TEST_CSV_LIST[@]} -eq 0 ]]; then
  echo "ERROR: No pred_test.csv found under:"
  echo "  - ${ATTO_ROOT}"
  echo "  - ${NANO_ROOT}"
  exit 1
fi

echo "Found TEST CSVs (${#TEST_CSV_LIST[@]}):"
printf ' - %s\n' "${TEST_CSV_LIST[@]}"

# ----------------------------
# Hybrid TEST ensemble (inner-join by path,label; mean prob)
# ----------------------------
ENSEMBLED_CSV="${OUT_DIR}/test_ens_labeled.csv"

python - <<'PY' "${ENSEMBLED_CSV}" "${#TEST_CSV_LIST[@]}" "${TEST_CSV_LIST[@]}"
import sys, os, pandas as pd
out_csv = sys.argv[1]
n = int(sys.argv[2]); files = sys.argv[3:3+n]

dfs=[]
for p in files:
    if not os.path.isfile(p): continue
    df = pd.read_csv(p)
    need = {"path","label","prob"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"{p} missing {miss}")
    df = df[["path","label","prob"]].copy()
    df["label"] = pd.to_numeric(df["label"], errors="raise").round().astype(int)
    key = os.path.basename(os.path.dirname(p))  # column per run
    df.rename(columns={"prob": key}, inplace=True)
    dfs.append(df)

if not dfs:
    raise SystemExit("No valid TEST CSVs")

ens = dfs[0]
for d in dfs[1:]:
    ens = ens.merge(d, on=["path","label"], how="inner")

seed_cols = [c for c in ens.columns if c not in ("path","label")]
ens["prob"] = ens[seed_cols].mean(axis=1)

ens = ens[["path","label","prob"]]
ens.to_csv(out_csv, index=False)
print(f"Hybrid TEST ensemble -> {out_csv} (N={len(ens)}, seeds={len(seed_cols)})")
PY

# ----------------------------
# Apply t* and save per-sample predictions
# ----------------------------
WITH_PRED_CSV="${OUT_DIR}/test_with_pred.csv"
python - <<'PY' "${ENSEMBLED_CSV}" "${WITH_PRED_CSV}" "${TSTAR}"
import sys, pandas as pd
src, dst, th = sys.argv[1], sys.argv[2], float(sys.argv[3])
df = pd.read_csv(src)
df["label"] = pd.to_numeric(df["label"], errors="raise").round().astype(int)
df["pred"] = (df["prob"] >= th).astype(int)
df.to_csv(dst, index=False)
print(f"Saved per-sample predictions -> {dst} (t*={th})")
PY

# ----------------------------
# Evaluate with fixed threshold (no optimization)
# ----------------------------
python -u "${EVAL_PY}" \
  --tag "test_ens_fixedth_hybrid" \
  --from_csv --csv "${ENSEMBLED_CSV}" \
  --threshold "${TSTAR}" \
  --per_domain \
  --plot \
  --out_dir "${OUT_DIR}"

echo "Artifacts in ${OUT_DIR}:"
echo " - ${ENSEMBLED_CSV}"
echo " - ${WITH_PRED_CSV}"
echo " - ${OUT_DIR}/metrics_test_ens_fixedth_hybrid.json"
echo " - ${OUT_DIR}/metrics_test_ens_fixedth_hybrid.csv"
echo " - PR/ROC/Calibration plots"
