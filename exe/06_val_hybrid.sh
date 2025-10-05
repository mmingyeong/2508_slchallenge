#!/bin/bash
#PBS -N sl_eval_threshold_val_ens_hybrid
#PBS -q long
#PBS -l nodes=1:ppn=2
#PBS -l mem=8gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -M mmingyeong@kasi.re.kr
#PBS -o sl_eval_threshold_val_ens_hybrid.pbs.out


PYENV="py312"

# ---- ATTO/NANO 결과 루트 (사용자 절대경로) ----
ATTO_ROOT="/home/users/mmingyeong/2508_slchallence/res"
NANO_ROOT="/home/users/mmingyeong/2508_slchallence/res/nano_ckpt"

# 산출물 저장 폴더 (ATTO 루트 쪽에 생성)
OUT_DIR="${ATTO_ROOT}/ensemble_hybrid_val"
mkdir -p "${OUT_DIR}"
ENSEMBLED_VAL_CSV="${OUT_DIR}/val_ens.csv"

# ----------------------------
# Environment
# ----------------------------
source ~/.bashrc
conda activate "${PYENV}"

echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "ATTO_ROOT = ${ATTO_ROOT}"
echo "NANO_ROOT = ${NANO_ROOT}"

# ----------------------------
# 1) atto + nano 의 pred_val.csv 수집
#    (신규 스타일 *_pred_val_pred_*/pred_val.csv 우선, 구형 pred_val/pred_val.csv 보조)
# ----------------------------
mapfile -t _FOUND < <(
  # ATTO (신규 스타일)
  ls -t "${ATTO_ROOT}"/_full_ckpt_fulltrain_s*_pred_val_pred_*/*/pred_val.csv 2>/dev/null || true
  ls -t "${ATTO_ROOT}"/_full_ckpt_fulltrain_s*_pred_val_pred_*/pred_val.csv        2>/dev/null || true
  # ATTO (구형: ckpt 디렉토리 내부 pred_val/pred_val.csv)
  ls -t "${ATTO_ROOT}"/_full_ckpt_fulltrain_s*/pred_val/pred_val.csv               2>/dev/null || true

  # NANO (신규 스타일)
  ls -t "${NANO_ROOT}"/_full_ckpt_fulltrain_nano_s*_pred_val_pred_*/*/pred_val.csv 2>/dev/null || true
  ls -t "${NANO_ROOT}"/_full_ckpt_fulltrain_nano_s*_pred_val_pred_*/pred_val.csv   2>/dev/null || true
  # NANO (구형: ckpt 디렉토리 내부 pred_val/pred_val.csv)
  ls -t "${NANO_ROOT}"/_full_ckpt_fulltrain_nano_s*/pred_val/pred_val.csv          2>/dev/null || true
)

# de-dup while preserving order
declare -A SEEN=()
CSV_LIST=()
for p in "${_FOUND[@]:-}"; do
  [[ -f "$p" ]] || continue
  if [[ -z "${SEEN[$p]:-}" ]]; then
    SEEN[$p]=1
    CSV_LIST+=("$p")
  fi
done

if [[ ${#CSV_LIST[@]} -eq 0 ]]; then
  echo "ERROR: No pred_val.csv found under:"
  echo "  - ${ATTO_ROOT}"
  echo "  - ${NANO_ROOT}"
  exit 1
fi

echo "Found VAL CSVs (${#CSV_LIST[@]}):"
printf ' - %s\n' "${CSV_LIST[@]}"

# ----------------------------
# 2) 하이브리드 VAL 앙상블 (path,label 교집합, 확률 평균)
# ----------------------------
python - <<'PY' "${ENSEMBLED_VAL_CSV}" "${#CSV_LIST[@]}" "${CSV_LIST[@]}"
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
    # 시드 키: 폴더명(충분히 유니크)
    seed = os.path.basename(os.path.dirname(p))
    df.rename(columns={"prob": seed}, inplace=True)
    dfs.append(df)

if not dfs:
    raise SystemExit("No valid VAL CSVs")

ens = dfs[0]
for d in dfs[1:]:
    ens = ens.merge(d, on=["path","label"], how="inner")  # 교집합 보존

seed_cols = [c for c in ens.columns if c not in ("path","label")]
ens["prob"] = ens[seed_cols].mean(axis=1)
ens = ens[["path","label","prob"]]
ens.to_csv(out_csv, index=False)
print(f"Hybrid VAL ensemble -> {out_csv} (N={len(ens)}, seeds={len(seed_cols)})")
PY

# ----------------------------
# 3) 임계값 최적화 (Youden's J)
#   evaluate.py 가 표준 파일명(metrics_val_ens.json)으로 저장한다고 가정
# ----------------------------
python -u "${ATTO_ROOT%/}/../${ATTO_ROOT##*/}/../" >/dev/null 2>&1 || true  # no-op; placeholder

python -u "${ATTO_ROOT%/}/../" >/dev/null 2>&1 || true  # no-op; placeholder

# 사용자가 쓰던 evaluate.py 경로가 /caefs/... 이었으면 아래 줄을 자신의 evaluate.py 절대경로로 맞추세요.
# 여기서는 atto/nano 모두 동일 소스일 것으로 보고 /home 경로 기준으로 수정하지 않고 호출만 유지합니다.
# 필요 시: EVAL_PY="/home/users/mmingyeong/2508_slchallence/src/evaluate.py"
EVAL_PY="/caefs/user/mmingyeong/2508_slchallence/src/evaluate.py"

python -u "${EVAL_PY}" \
  --tag "val_ens_hybrid" \
  --from_csv --csv "${ENSEMBLED_VAL_CSV}" \
  --optimize_threshold \
  --out_dir "${OUT_DIR}"

# 기존:
# METRICS_JSON="${OUT_DIR}/metrics_val_ens.json"

# 수정:
if   [[ -f "${OUT_DIR}/metrics_val_ens_hybrid.json" ]]; then
  METRICS_JSON="${OUT_DIR}/metrics_val_ens_hybrid.json"
elif [[ -f "${OUT_DIR}/metrics_val_ens.json" ]]; then
  METRICS_JSON="${OUT_DIR}/metrics_val_ens.json"
else
  echo "ERROR: metrics json not found in ${OUT_DIR}"
  exit 1
fi


# ----------------------------
# 4) t* 저장 (jq 있으면 사용, 없으면 파이썬)
# ----------------------------
if command -v jq >/dev/null 2>&1 && [[ -f "${METRICS_JSON}" ]]; then
  jq -r '.global.threshold' "${METRICS_JSON}" > "${OUT_DIR}/threshold.txt" || true
fi

if [[ ! -s "${OUT_DIR}/threshold.txt" ]]; then
  python - <<'PY' "${METRICS_JSON}" "${OUT_DIR}/threshold.txt"
import json, sys
mj, out = sys.argv[1], sys.argv[2]
with open(mj, "r") as f:
    m = json.load(f)
thr = None
try:
    thr = float(m.get("global", {}).get("threshold"))
except Exception:
    pass
if thr is None:
    # 0~1 범위 값 아무거나라도 찾기
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
with open(out, "w") as g:
    g.write(str(thr))
print(f"Saved threshold.txt (t*={thr})")
PY
fi

echo "Artifacts in ${OUT_DIR}:"
ls -lh "${OUT_DIR}"
