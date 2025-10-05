#!/bin/bash
#PBS -N sl_eval_labeled_test_with_tstar
#PBS -q long
#PBS -l nodes=1:ppn=2
#PBS -l mem=8gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -M mmingyeong@kasi.re.kr
#PBS -o sl_eval_labeled_test_with_tstar.pbs.out

# ============================
# User config
# ============================
PROJECT_ROOT="/caefs/user/mmingyeong/2508_slchallence"
PYENV="py312"

# (선택) 임계값 파일 직접 지정 시 우선 사용
# THRESHOLD_FILE="/home/users/mmingyeong/2508_slchallence/res/.../eval_val_ensemble/threshold.txt"

RUN_TAG="test_eval_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${PROJECT_ROOT}/res/nano_ckpt/test_eval_${RUN_TAG}"
mkdir -p "${OUT_DIR}"

# ============================
# Environment
# ============================
source ~/.bashrc
conda activate "${PYENV}"

echo "HOST: $(hostname)"
echo "DATE: $(date)"

# ============================
# 0) Load t* (threshold) without requiring threshold.txt
#     Priority: threshold.txt → metrics_val_ens.json → metrics_val_ens.csv → default 0.5
# ============================
# 검색 루트 확장: res/ 와 res/nano_ckpt/ 모두
SEARCH_ROOTS=(
  "${PROJECT_ROOT}/res"
  "${PROJECT_ROOT}/res/nano_ckpt"
)

# 0-1) threshold.txt 우선 사용(명시/자동)
if [[ -z "${THRESHOLD_FILE:-}" ]]; then
  for R in "${SEARCH_ROOTS[@]}"; do
    CAND=$(ls -t "${R}"/**/eval_val_ensemble/threshold.txt 2>/dev/null | head -n1 || true)
    if [[ -n "$CAND" ]]; then THRESHOLD_FILE="$CAND"; break; fi
  done
fi

if [[ -n "${THRESHOLD_FILE:-}" && -f "${THRESHOLD_FILE}" ]]; then
  TSTAR=$(cat "${THRESHOLD_FILE}")
  echo "Using t* from threshold.txt = ${TSTAR}  (${THRESHOLD_FILE})"
else
  echo "threshold.txt not found; trying metrics files..."

  # 0-2) metrics_val_ens.json에서 threshold 추출
  JSON_METRICS=""
  for R in "${SEARCH_ROOTS[@]}"; do
    CAND=$(ls -t "${R}"/**/eval_val_ensemble/metrics_val_ens.json 2>/dev/null | head -n1 || true)
    if [[ -n "$CAND" ]]; then JSON_METRICS="$CAND"; break; fi
  done

  if [[ -n "$JSON_METRICS" && -f "$JSON_METRICS" ]]; then
    TSTAR=$(
    python - <<'PY' "$JSON_METRICS"
import sys, json
p = sys.argv[1]
with open(p, "r") as f:
    m = json.load(f)
# 후보 키들 시도
candidates = [
    ("global","threshold"),
    ("threshold",),
    ("optimal_threshold",),
    ("youden_threshold",),
]
val = None
for path in candidates:
    cur = m
    try:
        for k in path:
            cur = cur[k]
        val = float(cur)
        break
    except Exception:
        pass
if val is None:
    # 평면 키 탐색(0~1 범위)
    def find_num(d):
        if isinstance(d, dict):
            for k,v in d.items():
                r = find_num(v)
                if r is not None: return r
        elif isinstance(d, (int,float)) and 0.0 <= d <= 1.0:
            return float(d)
        return None
    val = find_num(m)
if val is None:
    raise SystemExit("NO_THRESHOLD")
print(val)
PY
    ) || true
    if [[ -n "$TSTAR" ]]; then
      echo "Using t* from metrics_val_ens.json = ${TSTAR}  (${JSON_METRICS})"
    fi
  fi

  # 0-3) CSV에서 threshold 추출
  if [[ -z "${TSTAR:-}" ]]; then
    CSV_METRICS=""
    for R in "${SEARCH_ROOTS[@]}"; do
      CAND=$(ls -t "${R}"/**/eval_val_ensemble/metrics_val_ens.csv 2>/dev/null | head -n1 || true)
      if [[ -n "$CAND" ]]; then CSV_METRICS="$CAND"; break; fi
    done
    if [[ -n "$CSV_METRICS" && -f "$CSV_METRICS" ]]; then
      TSTAR=$(
      python - <<'PY' "$CSV_METRICS"
import sys, pandas as pd
p = sys.argv[1]
df = pd.read_csv(p)
# threshold / youden / tstar 등 열 이름 탐색
for col in df.columns:
    if str(col).strip().lower() in {"threshold","tstar","t_star","youden_threshold"}:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        s = s[(s>=0.0) & (s<=1.0)]
        if len(s): 
            # 'global' 행이 있으면 우선 사용
            if "scope" in df.columns:
                g = df.loc[df["scope"].astype(str).str.lower().eq("global"), col]
                g = pd.to_numeric(g, errors="coerce").dropna()
                if len(g): print(float(g.iloc[0])); raise SystemExit
            print(float(s.iloc[0])); raise SystemExit
# 전열 스캔(0~1 값)
for col in df.columns:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    s = s[(s>=0.0) & (s<=1.0)]
    if len(s):
        print(float(s.iloc[0])); raise SystemExit
raise SystemExit("NO_THRESHOLD")
PY
      ) || true
      if [[ -n "$TSTAR" ]]; then
        echo "Using t* from metrics_val_ens.csv = ${TSTAR}  (${CSV_METRICS})"
      fi
    fi
  fi

  # 0-4) 최후: 기본값 0.5
  if [[ -z "${TSTAR:-}" ]]; then
    TSTAR="0.5"
    echo "WARN: Could not locate threshold in any metrics. Falling back to default t* = ${TSTAR}"
  fi
fi

# ============================
# 1) Collect per-seed TEST predictions (labeled)
# ============================
if [[ ${#TEST_CSV_LIST[@]:-0} -eq 0 ]]; then
  mapfile -t TEST_CSV_LIST < <(ls -t "${PROJECT_ROOT}"/res/nano_ckpt/*_pred_test_*/pred_test.csv 2>/dev/null || true)
  if [[ ${#TEST_CSV_LIST[@]} -eq 0 ]]; then
    mapfile -t TEST_CSV_LIST < <(ls -t "${PROJECT_ROOT}"/res/nano_ckpt/_full_ckpt_fulltrain_nano_*/pred_test/pred_test.csv 2>/dev/null || true)
  fi
fi

if [[ ${#TEST_CSV_LIST[@]} -eq 0 ]]; then
  echo "ERROR: TEST용 pred_test.csv 파일들을 찾지 못했습니다. 먼저 predict.py --which test를 실행하세요."
  exit 1
fi

echo "Found TEST CSVs:"
printf ' - %s\n' "${TEST_CSV_LIST[@]}"

# ============================
# 2) Ensemble (label 포함, path 기준 inner-join 평균)
# ============================
ENSEMBLED_CSV="${OUT_DIR}/test_ens_labeled.csv"

python - <<'PY' "${ENSEMBLED_CSV}" "${#TEST_CSV_LIST[@]}" "${TEST_CSV_LIST[@]}"
import sys, os, pandas as pd
out_csv = sys.argv[1]
n = int(sys.argv[2])
files = sys.argv[3:3+n]
if n == 0:
    raise SystemExit("No TEST CSVs provided")

dfs = []
for f in files:
    if not os.path.isfile(f): continue
    df = pd.read_csv(f)
    need = {"path","label","prob"}
    miss = need - set(df.columns)
    if miss: raise SystemExit(f"{f} missing columns: {miss}")
    df = df[["path","label","prob"]].copy()
    df["label"] = pd.to_numeric(df["label"], errors="raise").round().astype(int)
    tag = os.path.basename(os.path.dirname(f))
    df.rename(columns={"prob": tag}, inplace=True)
    dfs.append(df)

if not dfs:
    raise SystemExit("No valid TEST CSVs to ensemble")

ens = dfs[0]
for d in dfs[1:]:
    ens = ens.merge(d, on=["path","label"], how="inner")

seed_cols = [c for c in ens.columns if c not in ("path","label")]
ens["prob"] = ens[seed_cols].mean(axis=1)
ens = ens[["path","label","prob"]]
ens.to_csv(out_csv, index=False)
print(f"Ensembled TEST CSV -> {out_csv} (N={len(ens)}, seeds={len(seed_cols)})")
PY

# ============================
# 3) Apply fixed threshold t*
# ============================
WITH_PRED_CSV="${OUT_DIR}/test_with_pred.csv"
python - <<'PY' "${ENSEMBLED_CSV}" "${WITH_PRED_CSV}" "${TSTAR}"
import sys, pandas as pd
src, dst, th = sys.argv[1], sys.argv[2], float(sys.argv[3])
df = pd.read_csv(src)
df["label"] = pd.to_numeric(df["label"], errors="raise").round().astype(int)
df["pred"] = (df["prob"] >= th).astype(int)
df.to_csv(dst, index=False)
print(f"Saved per-sample preds -> {dst}  (threshold={th:.6f}, N={len(df)})")
PY

# ============================
# 4) Evaluate with fixed threshold (no optimization)
# ============================
python -u "${PROJECT_ROOT}/src/evaluate.py" \
  --tag "test_ens_fixedth" \
  --from_csv --csv "${ENSEMBLED_CSV}" \
  --threshold "${TSTAR}" \
  --per_domain \
  --plot \
  --out_dir "${OUT_DIR}"

echo "Artifacts:"
echo " - ${ENSEMBLED_CSV}"
echo " - ${WITH_PRED_CSV}"
echo " - ${OUT_DIR}/metrics_test_ens_fixedth.json"
echo " - ${OUT_DIR}/metrics_test_ens_fixedth.csv"
echo " - ${OUT_DIR}/roc_test_ens_fixedth.png, pr_*.png, calibration_*.png"
echo "DONE at $(date)"
