#!/bin/bash
#PBS -N sl_predict_and_submit_nano_final
#PBS -q long
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l mem=32gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -M mmingyeong@kasi.re.kr
#PBS -o /home/users/mmingyeong/2508_slchallence/final/sl_predict_and_submit_nano_final.pbs.out

# =========================
# User config
# =========================
PYENV="py312"

# 경로 설정
TEST_DIR="/caefs/data/IllustrisTNG/slchallenge/test_dataset_updated/test_dataset_updated"
NANO_ROOT="/home/users/mmingyeong/2508_slchallence/res/nano_ckpt"
PROJECT_ROOT="/caefs/user/mmingyeong/2508_slchallence"
PREDICT_PY="${PROJECT_ROOT}/src/predict.py"
FINAL_DIR="/home/users/mmingyeong/2508_slchallence/final"

# 시드별 VAL metrics 경로
NANO_METRICS=(
  "/home/users/mmingyeong/2508_slchallence/res/nano_ckpt/_full_ckpt_fulltrain_nano_s42_20250925_184205_pred_val_pred_20251005_132054/eval_val_ensemble/metrics_val_ens.json"
  "/home/users/mmingyeong/2508_slchallence/res/nano_ckpt/_full_ckpt_fulltrain_nano_s101_20250925_184206_pred_val_pred_20251005_132054/eval_val_ensemble/metrics_val_ens.json"
  "/home/users/mmingyeong/2508_slchallence/res/nano_ckpt/_full_ckpt_fulltrain_nano_s202_20250927_134111_pred_val_pred_20251005_132054/eval_val_ensemble/metrics_val_ens.json"
  "/home/users/mmingyeong/2508_slchallence/res/nano_ckpt/_full_ckpt_fulltrain_nano_s303_20250927_191302_pred_val_pred_20251005_132054/eval_val_ensemble/metrics_val_ens.json"
  "/home/users/mmingyeong/2508_slchallence/res/nano_ckpt/_full_ckpt_fulltrain_nano_s404_20250929_150029_pred_val_pred_20251005_132054/eval_val_ensemble/metrics_val_ens.json"
)
THR_AGG="mean"

# 제출 파일 경로
SUBMIT_CSV="${FINAL_DIR}/submission_nano_ensemble.csv"

# 예측 파라미터
BATCH=128
NUM_WORKERS=8
DROP_PATH=0.05
DEVICE="cuda"
APPLY_NORM="--apply_normalization"
SMOOTHING="--smoothing_mode gaussian --gaussian_sigma 1.0"

# =========================
# Environment
# =========================
source ~/.bashrc
conda activate "${PYENV}"

RUN_TAG="finaltest_$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="${FINAL_DIR}/run_predict_final_${RUN_TAG}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "############################################"
echo "# Strong Lensing Final Prediction (Nano)   #"
echo "############################################"
echo "HOST: $(hostname)"
echo "DATE: $(date)"
nvidia-smi || true
echo "TEST_DIR = ${TEST_DIR}"
echo "NANO_ROOT= ${NANO_ROOT}"
echo "FINAL_DIR= ${FINAL_DIR}"
echo "MAIN LOG = ${MAIN_LOG}"

# =========================
# 0) 테스트셋 '목록(Manifest)' 4개 생성 (심볼릭 링크 생성 제거)
#    - 단일 파이썬 프로세스로 4-way 해시 분배 (빠르고 안정적)
# =========================
TMP_ROOT="${NANO_ROOT}/_tmp_finaltest_inputs_${RUN_TAG}"
MAN_DIR="${TMP_ROOT}/manifests"
mkdir -p "${MAN_DIR}"

LIST_LENS="${MAN_DIR}/lens.list"
LIST_NLENS="${MAN_DIR}/nlens.list"
LIST_HSC_LENS="${MAN_DIR}/hsc_lens.list"
LIST_HSC_NLENS="${MAN_DIR}/hsc_nlens.list"

: > "${LIST_LENS}"
: > "${LIST_NLENS}"
: > "${LIST_HSC_LENS}"
: > "${LIST_HSC_NLENS}"

echo "[INFO] Building 4 manifests from ${TEST_DIR} (single Python process)..."
python - <<'PY' "${TEST_DIR}" "${LIST_LENS}" "${LIST_NLENS}" "${LIST_HSC_LENS}" "${LIST_HSC_NLENS}"
import os, sys, hashlib
test_dir, out0, out1, out2, out3 = sys.argv[1:6]
total = 0
# 큰 버퍼로 I/O 효율 향상
with open(out0, 'w', buffering=1<<20) as f0, \
     open(out1, 'w', buffering=1<<20) as f1, \
     open(out2, 'w', buffering=1<<20) as f2, \
     open(out3, 'w', buffering=1<<20) as f3:
    with os.scandir(test_dir) as it:
        for ent in it:
            if not ent.is_file():
                continue
            name = ent.name
            if not name.endswith('.fits'):
                continue
            # 파일 내용은 읽지 않고 파일명만 해시
            h = int(hashlib.md5(name.encode('utf-8')).hexdigest()[:8], 16) % 4
            fp = os.path.join(test_dir, name)
            (f0 if h==0 else f1 if h==1 else f2 if h==2 else f3).write(fp + '\n')
            total += 1
            if total % 100000 == 0:
                print(f"[INFO] manifest builder processed {total} files", flush=True)
print(f"[OK] manifest builder total: {total}")
PY

CNT0=$(wc -l < "${LIST_LENS}")
CNT1=$(wc -l < "${LIST_NLENS}")
CNT2=$(wc -l < "${LIST_HSC_LENS}")
CNT3=$(wc -l < "${LIST_HSC_NLENS}")
TOTAL=$((CNT0 + CNT1 + CNT2 + CNT3))

echo "[INFO] Manifest counts: LENS=${CNT0}, NLENS=${CNT1}, HSC_LENS=${CNT2}, HSC_NLENS=${CNT3}, TOTAL=${TOTAL}"

if [[ "${TOTAL}" -eq 0 ]]; then
  echo "ERROR: No .fits discovered in ${TEST_DIR}"
  exit 1
fi

# =========================
# 1) 체크포인트 디렉터리 탐색
# =========================
mapfile -t CKPT_DIRS < <(
  find "${NANO_ROOT}" -maxdepth 1 -type d -name "_full_ckpt_fulltrain_nano_*" \
    -not -name "*_pred_*" -not -name "*_eval_*" | sort -V
)
if [[ ${#CKPT_DIRS[@]} -eq 0 ]]; then
  echo "ERROR: no NANO ckpt dirs found."
  exit 1
fi

# =========================
# 2) 시드별 예측 실행 (디렉토리 대신 '목록' 옵션 사용)
#     predict.py가 아래 *_list 인자를 지원해야 합니다.
#     (예: --slsim_lenses_list path/to/list.txt)
# =========================
for CKPT_DIR in "${CKPT_DIRS[@]}"; do
  MODEL_PATH="${CKPT_DIR}/best.pt"
  [[ -f "${MODEL_PATH}" ]] || MODEL_PATH="${CKPT_DIR}/last.pt"
  [[ -f "${MODEL_PATH}" ]] || { echo "SKIP: no model in ${CKPT_DIR}"; continue; }

  CKPT_TAG="$(basename "${CKPT_DIR}")"
  OUT_DIR="${CKPT_DIR}_pred_${RUN_TAG}"
  LOG_FILE="${FINAL_DIR}/predict_${CKPT_TAG}_${RUN_TAG}.log"
  CSV_OUT="${OUT_DIR}/pred_test.csv"
  mkdir -p "${OUT_DIR}"

  echo ">>> Predicting TEST | ${CKPT_TAG}"
  echo "    MODEL: ${MODEL_PATH}"
  echo "    OUT:   ${OUT_DIR}"
  echo "    LOG:   ${LOG_FILE}"

  python -u "${PREDICT_PY}" \
    --slsim_lenses_list "${LIST_LENS}" \
    --slsim_nonlenses_list "${LIST_NLENS}" \
    --hsc_lenses_list "${LIST_HSC_LENS}" \
    --hsc_nonlenses_list "${LIST_HSC_NLENS}" \
    --which test \
    --batch_size "${BATCH}" \
    --num_workers "${NUM_WORKERS}" \
    --train_frac 0.01 --val_frac 0.01 --test_frac 0.98 \
    --model_path "${MODEL_PATH}" \
    --model_size nano \
    --drop_path "${DROP_PATH}" \
    --device "${DEVICE}" \
    --output_dir "${OUT_DIR}" \
    ${APPLY_NORM} ${SMOOTHING} \
    >"${LOG_FILE}" 2>&1

  if [[ -f "${CSV_OUT}" ]]; then
    echo "[OK] Prediction done: ${CSV_OUT}"
  else
    echo "[WARN] Missing pred CSV for ${CKPT_TAG}"
  fi
done

# =========================
# 3) VAL threshold (t*) 계산
# =========================
TSTAR=$(
python - <<'PY' "${THR_AGG}" "${#NANO_METRICS[@]}" "${NANO_METRICS[@]}"
import sys, json, numpy as np
agg=sys.argv[1]; n=int(sys.argv[2]); files=sys.argv[3:3+n]
vals=[]
for p in files:
    try:
        with open(p) as f: m=json.load(f)
        if "global" in m and "threshold" in m["global"]: vals.append(float(m["global"]["threshold"]))
        elif "threshold" in m: vals.append(float(m["threshold"]))
    except: pass
print( np.mean(vals) if (vals and agg=="mean") else (np.median(vals) if vals else 0.5) )
PY
)
echo "[INFO] Using VAL threshold t* = ${TSTAR}"

# =========================
# 4) Ensemble + 제출 CSV 생성
# =========================
python - <<'PY' "${TEST_DIR}" "${NANO_ROOT}" "${RUN_TAG}" "${TSTAR}" "${SUBMIT_CSV}"
import sys, os, glob, numpy as np, pandas as pd
test_dir, nano_root, run_tag, tstar, out_csv = sys.argv[1:6]
tstar=float(tstar)

# 테스트 id 추출 (디렉토리 스캔은 한 번만)
fits=sorted(glob.glob(os.path.join(test_dir, "*.fits")))
ids=[os.path.splitext(os.path.basename(p))[0] for p in fits]
ens=pd.DataFrame({"id":ids})

# 각 시드의 pred_test.csv 로부터 prob 평균 앙상블
paths=sorted(glob.glob(os.path.join(nano_root,f"_full_ckpt_fulltrain_nano_*_pred_{run_tag}","pred_test.csv")))
if not paths: raise SystemExit("No prediction CSVs found.")

for p in paths:
    df=pd.read_csv(p)
    id_col=next((c for c in ["path","file","id","ID","name"] if c in df.columns),None)
    if id_col is None:
        raise SystemExit(f"No id-like column in {p}")
    ids_col=df[id_col].astype(str).map(lambda x: os.path.splitext(os.path.basename(x))[0])
    if "prob" not in df.columns:
        raise SystemExit(f"No 'prob' column in {p}")
    tag=os.path.basename(os.path.dirname(p))
    ens=ens.merge(pd.DataFrame({"id":ids_col, tag:df["prob"]}),on="id",how="left")

ens["prob"]=ens.drop(columns=["id"]).mean(axis=1)
ens["preds"]=(ens["prob"]>=tstar).astype(int)

cols=["id","preds"]+[f for f in [
"RA","Dec","zlens","vel disp","ell_m","ell_m_PA","sh","sh_PA","Rein",
"ell_l","ell_l_PA","Reff_l","n_l_sers",
"mag_lens_g","mag_lens_r","mag_lens_i","mag_lens_z","mag_lens_y",
"zsrc","srcx","srcy",
"mag_src_g","mag_src_r","mag_src_i","mag_src_z","mag_src_y",
"ell_s","ell_s_PA","Reff_s","n_s_sers"]]
sub=pd.DataFrame(index=ens.index,columns=cols)
sub["id"]=ens["id"].astype(str)
sub["preds"]=ens["preds"].astype(int)
for c in cols:
    if c not in ("id","preds"): sub[c]=-99

os.makedirs(os.path.dirname(out_csv),exist_ok=True)
sub.to_csv(out_csv,index=False)
print(f"[OK] submission saved -> {out_csv}")
print(sub.head(5))
PY

echo "[INFO] All done. Logs saved to ${FINAL_DIR}"
echo "[INFO] Manifests kept at: ${MAN_DIR}"
echo "DONE at $(date)"
