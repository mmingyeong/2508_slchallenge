cat > clean_submission.py <<'PY'
# -*- coding: utf-8 -*-
import csv, re, unicodedata, io

SRC = "submission_nano_ensemble.csv"
DST = "submission_clean.csv"

def clean_id(raw: str) -> str:
    s = raw.strip().strip('\ufeff')
    s = unicodedata.normalize("NFC", s)
    m = re.match(r'^(object_\d{1,})(?:_[a-zA-Z]+)?$', s)
    if m:
        return m.group(1)
    if s.startswith("object_"):
        parts = s.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
    return s

def clean_pred(raw: str) -> int:
    s = re.sub(r'[^0-9+-]', '', raw.strip())
    v = int(s) if s else 0
    return 1 if v >= 1 else 0

with io.open(SRC, "r", encoding="utf-8", newline="") as f:
    text = f.read().lstrip('\ufeff')

rows = []
for line in text.splitlines():
    line = line.strip()
    if not line:
        continue
    parts = line.split(",")
    if parts and parts[0].strip().lower() == "id":
        continue
    if len(parts) < 2:
        continue
    rows.append((clean_id(parts[0]), clean_pred(parts[1])))

dedup = {}
for k, v in rows:
    dedup[k] = v
final_rows = sorted(dedup.items(), key=lambda kv: kv[0])

with io.open(DST, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "preds"])
    for k, v in final_rows:
        w.writerow([k, v])

print(f"[OK] Wrote {len(final_rows)} rows to {DST}")
PY

python3 clean_submission.py
