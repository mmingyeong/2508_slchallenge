#!/bin/bash
# ============================================
# make_prob_submission.sh
# - Strong Lensing Challenge probability-only CSV 생성
# - 기존 pred_test.csv 파일에서 id, prob만 모아서 제출용 CSV 생성
# ============================================

TEST_DIR="/caefs/data/IllustrisTNG/slchallenge/test_dataset_updated/test_dataset_updated"
NANO_ROOT="/home/users/mmingyeong/2508_slchallence/res/nano_ckpt"
RUN_TAG="finaltest_20251014_150833"
OUT_CSV="/home/users/mmingyeong/2508_slchallence/final/submission_nano_probs.csv"

echo "[INFO] Building probability submission..."
echo "TEST_DIR = $TEST_DIR"
echo "NANO_ROOT= $NANO_ROOT"
echo "RUN_TAG   = $RUN_TAG"
echo "OUT_CSV   = $OUT_CSV"

python - <<PY "$TEST_DIR" "$NANO_ROOT" "$RUN_TAG" "$OUT_CSV"
import sys, os, glob
import pandas as pd

test_dir, nano_root, run_tag, out_csv = sys.argv[1:5]

# 1) 테스트 id 리스트
fits_paths = sorted(glob.glob(os.path.join(test_dir, "*.fits")))
if not fits_paths:
    raise SystemExit(f"No .fits files found in {test_dir}")

ids = [os.path.splitext(os.path.basename(p))[0] for p in fits_paths]
ens = pd.DataFrame({"id": ids})
print(f"[INFO] Found {len(ids)} test images.")

# 2) 시드별 pred_test.csv 수집
pattern = os.path.join(
    nano_root, f"_full_ckpt_fulltrain_nano_*_pred_{run_tag}", "pred_test.csv"
)
pred_csvs = sorted(glob.glob(pattern))
if not pred_csvs:
    raise SystemExit(f"No pred_test.csv found with pattern: {pattern}")

print("[INFO] Using pred_test.csv files:")
for p in pred_csvs:
    print("  -", p)

# 3) 각 시드의 prob 컬럼 병합
for p in pred_csvs:
    df = pd.read_csv(p)

    id_col = next((c for c in ["path","file","id","ID","name"] if c in df.columns), None)
    if id_col is None:
        raise SystemExit(f"No id-like column in {p}")

    if "prob" not in df.columns:
        raise SystemExit(f"No 'prob' column in {p}")

    ids_col = df[id_col].astype(str).map(
        lambda x: os.path.splitext(os.path.basename(x))[0]
    )
    seed_tag = os.path.basename(os.path.dirname(p))

    tmp = pd.DataFrame({"id": ids_col, seed_tag: df["prob"].astype(float)})
    ens = ens.merge(tmp, on="id", how="left")

# 4) 시드 평균 → 최종 prob
seed_cols = [c for c in ens.columns if c != "id"]
ens["prob"] = ens[seed_cols].mean(axis=1)

# 5) 최종 제출 형식: id, prob
sub = ens[["id", "prob"]].copy()
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
sub.to_csv(out_csv, index=False)

print(f"[OK] Probability submission saved -> {out_csv}")
print(sub.head())

PY

echo "[DONE] All completed successfully."
