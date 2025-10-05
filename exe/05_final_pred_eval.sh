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

# (선택) 임계값 파일 지정. 비우면 res/eval_val_*/threshold.txt 중 가장 최근 파일 자동 탐색
# THRESHOLD_FILE="${PROJECT_ROOT}/res/eval_val_ensemble/threshold.txt"

# (선택) 사용할 테스트 예측 CSV들을 직접 지정하고 싶다면 아래 배열에 나열
# TEST_CSV_LIST=(
#   "/.../_full_ckpt_s42_.../pred_test/pred_test.csv"
#   "/.../_full_ckpt_s101_.../pred_test/pred_test.csv"
# )

# 결과 출력 루트
RUN_TAG="test_eval_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${PROJECT_ROOT}/res/test_eval_${RUN_TAG}"
mkdir -p "${OUT_DIR}"

# ============================
# Environment
# ============================
source ~/.bashrc
conda activate "${PYENV}"

echo "HOST: $(hostname)"
echo "DATE: $(date)"

# ============================
# 0) Load t* (threshold)
# ============================
if [[ -z "${THRESHOLD_FILE:-}" ]]; then
  THRESHOLD_FILE=$(ls -t "${PROJECT_ROOT}"/res/eval_val_*/threshold.txt 2>/dev/null | head -n1 || true)
fi
if [[ -z "${THRESHOLD_FILE}" || ! -f "${THRESHOLD_FILE}" ]]; then
  echo "ERROR: threshold.txt를 찾지 못했습니다. THRESHOLD_FILE을 지정하거나 VAL 임계값 산출을 먼저 수행하세요."
  exit 1
fi
TSTAR=$(cat "${THRESHOLD_FILE}")
echo "Using fixed threshold t* = ${TSTAR}  (from ${THRESHOLD_FILE})"

# ============================
# 1) Collect per-seed TEST predictions (labeled)
#     기대 컬럼: path, domain(옵션), label, prob, pred
# ============================
if [[ ${#TEST_CSV_LIST[@]:-0} -eq 0 ]]; then
  # 새 스타일: .../res/<CKPT_TAG>_pred_test_<RUN_TAG>/pred_test.csv
  mapfile -t TEST_CSV_LIST < <(ls -t "${PROJECT_ROOT}"/res/*_pred_test_*/pred_test.csv 2>/dev/null || true)
  # 구 스타일(있을 경우): .../res/_full_ckpt_*/pred_test/pred_test.csv
  if [[ ${#TEST_CSV_LIST[@]} -eq 0 ]]; then
    mapfile -t TEST_CSV_LIST < <(ls -t "${PROJECT_ROOT}"/res/_full_ckpt_*/pred_test/pred_test.csv 2>/dev/null || true)
  fi
fi

if [[ ${#TEST_CSV_LIST[@]} -eq 0 ]]; then
  echo "ERROR: TEST용 pred_test.csv 파일들을 찾지 못했습니다. 먼저 predict.py --which test를 시드별로 실행하세요."
  exit 1
fi

echo "Found TEST CSVs:"
printf ' - %s\n' "${TEST_CSV_LIST[@]}"

# ============================
# 2) Ensemble (label 포함, path 기준 inner-join 평균)
#     출력: test_ens_labeled.csv  (columns: path, label, prob)
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
    if not os.path.isfile(f):
        continue
    df = pd.read_csv(f)
    need = {"path","label","prob"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{f} missing columns: {missing}")
    # 병합 키: path, label (라벨 있는 내부 테스트셋 전제)
    df = df[["path","label","prob"]].copy()
    # 열 이름: 이 파일의 부모폴더명을 사용(유일 식별자)
    key = os.path.basename(os.path.dirname(f))
    df.rename(columns={"prob": key}, inplace=True)
    dfs.append(df)

if not dfs:
    raise SystemExit("No valid TEST CSVs to ensemble")

ens = dfs[0]
for d in dfs[1:]:
    ens = ens.merge(d, on=["path","label"], how="inner")

seed_cols = [c for c in ens.columns if c not in ("path","label")]
ens["prob"] = ens[seed_cols].mean(axis=1)  # 동등가중 평균
ens = ens[["path","label","prob"]]
ens.to_csv(out_csv, index=False)
print(f"Ensembled TEST CSV -> {out_csv} (N={len(ens)}, seeds={len(seed_cols)})")
PY

# ============================
# 3) Apply fixed threshold t* → per-sample preds CSV (분석용)
#     출력: test_with_pred.csv  (path, label, prob, pred)
# ============================
WITH_PRED_CSV="${OUT_DIR}/test_with_pred.csv"
python - <<'PY' "${ENSEMBLED_CSV}" "${WITH_PRED_CSV}" "${TSTAR}"
import sys, pandas as pd
src, dst, th = sys.argv[1], sys.argv[2], float(sys.argv[3])
df = pd.read_csv(src)
df["pred"] = (df["prob"] >= th).astype(int)
df.to_csv(dst, index=False)
print(f"Saved per-sample preds -> {dst}  (threshold={th:.6f}, N={len(df)})")
PY

# ============================
# 4) Evaluate with fixed threshold (no optimization)
#     evaluate.py는 CSV의 'pred' 열이 있어도 무시하고,
#     우리가 지정한 --threshold로 재이진화하여 지표를 계산합니다.
# ============================
python -u "${PROJECT_ROOT}/src/evaluate.py" \
  --tag "test_ens_fixedth" \
  --from_csv --csv "${ENSEMBLED_CSV}" \
  --threshold "${TSTAR}" \
  --per_domain \
  --plot \
  --out_dir "${OUT_DIR}"

echo "Artifacts:"
echo " - ${ENSEMBLED_CSV}         # label 포함 앙상블 확률"
echo " - ${WITH_PRED_CSV}         # t* 적용 per-sample 예측"
echo " - ${OUT_DIR}/metrics_test_ens_fixedth.json"
echo " - ${OUT_DIR}/metrics_test_ens_fixedth.csv"
echo " - ${OUT_DIR}/roc_test_ens_fixedth.png, pr_*.png, calibration_*.png"
echo "DONE at $(date)"
