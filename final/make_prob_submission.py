#!/usr/bin/env python
# make_prob_submission.py
#
# 기존 NANO 시드별 pred_test.csv들에서 확률(prob)만 모아서
# id, prob 형식의 최종 submission CSV를 만드는 스크립트.

import os
import glob
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Assemble probability-only submission from pred_test.csv files."
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Directory containing test .fits files (to define id list).",
    )
    parser.add_argument(
        "--nano_root",
        type=str,
        required=True,
        help="Root directory where _full_ckpt_fulltrain_nano_* folders live.",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        required=True,
        help="Tag used when running predict.py (e.g., finaltest_20251120_123456).",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Path to save the probability submission CSV (id, prob).",
    )

    args = parser.parse_args()

    test_dir = args.test_dir
    nano_root = args.nano_root
    run_tag = args.run_tag
    out_csv = args.out_csv

    # 1) 테스트셋에서 id 목록 만들기 (파일명에서 .fits 제거)
    fits_paths = sorted(glob.glob(os.path.join(test_dir, "*.fits")))
    if not fits_paths:
        raise SystemExit(f"No .fits files found in {test_dir}")

    ids = [os.path.splitext(os.path.basename(p))[0] for p in fits_paths]
    ens = pd.DataFrame({"id": ids})
    print(f"[INFO] Found {len(ids)} test images.")

    # 2) 시드별 pred_test.csv 찾기
    pattern = os.path.join(
        nano_root,
        f"_full_ckpt_fulltrain_nano_*_pred_{run_tag}",
        "pred_test.csv",
    )
    pred_csvs = sorted(glob.glob(pattern))

    if not pred_csvs:
        raise SystemExit(f"No pred_test.csv found with pattern: {pattern}")

    print("[INFO] Using pred_test.csv files:")
    for p in pred_csvs:
        print("  -", p)

    # 3) 각 pred_test.csv에서 prob 컬럼만 병합 (id는 파일명에서 다시 추출)
    for p in pred_csvs:
        df = pd.read_csv(p)

        # id로 쓸 수 있는 컬럼 찾기 (path, file, id 등)
        id_col = None
        for cand in ["path", "file", "id", "ID", "name"]:
            if cand in df.columns:
                id_col = cand
                break
        if id_col is None:
            raise SystemExit(f"[ERROR] No id-like column in {p}")

        if "prob" not in df.columns:
            raise SystemExit(f"[ERROR] No 'prob' column in {p}")

        # id: 파일 basename에서 .fits 제거
        ids_col = df[id_col].astype(str).map(
            lambda x: os.path.splitext(os.path.basename(x))[0]
        )
        seed_tag = os.path.basename(os.path.dirname(p))

        tmp = pd.DataFrame({"id": ids_col, seed_tag: df["prob"].astype(float)})
        ens = ens.merge(tmp, on="id", how="left")

    # 4) 시드별 확률을 평균 내어 최종 prob 생성
    seed_cols = [c for c in ens.columns if c != "id"]
    ens["prob"] = ens[seed_cols].mean(axis=1)

    # NaN 체크 (혹시 빠진 경우)
    if ens["prob"].isna().any():
        n_nan = ens["prob"].isna().sum()
        print(f"[WARN] {n_nan} entries have NaN probability (missing from some CSVs).")

    # 5) 최종 제출용: id, prob
    sub = ens[["id", "prob"]].copy()
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    sub.to_csv(out_csv, index=False)

    print(f"[OK] Saved probability submission -> {out_csv}")
    print(sub.head())


if __name__ == "__main__":
    main()
