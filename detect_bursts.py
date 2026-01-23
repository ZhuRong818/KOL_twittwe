import json
import pandas as pd
from datetime import timedelta

# ========= 配置 =========
INPUT_FILE = "KOL.jsonl"
TIME_GRAIN = "1H"          # 1 hour
BASELINE_HOURS = 168       # 7 days
Z_THRESHOLD = 3.0
MIN_COUNT = 3
MERGE_GAP_HOURS = 2
# =======================


def parse_created_at(s):
    # "Tue Oct 07 12:38:00 +0000 2025"
    return pd.to_datetime(s, utc=True)


# ---------- 1. 读入数据 ----------
rows = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        rows.append({
            "kol": obj["kol_username"],
            "created_at": parse_created_at(obj["tweet"]["createdAt"])
        })

df = pd.DataFrame(rows)
df = df.sort_values("created_at")


# ---------- 2. 按 KOL + 小时聚合 ----------
bursts = []

for kol, g in df.groupby("kol"):
    g = g.set_index("created_at")

    counts = (
        g
        .resample(TIME_GRAIN)
        .size()
        .rename("count")
        .to_frame()
    )

    # rolling baseline
    counts["mu"] = counts["count"].rolling(BASELINE_HOURS).mean()
    counts["sigma"] = counts["count"].rolling(BASELINE_HOURS).std()

    counts["z"] = (counts["count"] - counts["mu"]) / (counts["sigma"] + 1e-6)

    # burst condition
    counts["is_burst"] = (
        (counts["z"] >= Z_THRESHOLD) &
        (counts["count"] >= MIN_COUNT)
    )

    # ---------- 3. 合并连续 burst ----------
    active = None
    for t, row in counts.iterrows():
        if row["is_burst"]:
            if active is None:
                active = {
                    "kol": kol,
                    "start": t,
                    "end": t,
                    "peak_count": row["count"]
                }
            else:
                gap = (t - active["end"]) / timedelta(hours=1)
                if gap <= MERGE_GAP_HOURS:
                    active["end"] = t
                    active["peak_count"] = max(active["peak_count"], row["count"])
                else:
                    bursts.append(active)
                    active = {
                        "kol": kol,
                        "start": t,
                        "end": t,
                        "peak_count": row["count"]
                    }
        else:
            if active is not None:
                bursts.append(active)
                active = None

    if active is not None:
        bursts.append(active)


# ---------- 4. 输出 ----------
burst_df = pd.DataFrame(bursts)
burst_df["duration_hours"] = (
    (burst_df["end"] - burst_df["start"]) / timedelta(hours=1) + 1
)

burst_df = burst_df.sort_values(["kol", "start"])

burst_df.to_csv("burst_periods.csv", index=False)

print(" Burst detection done.")
print(burst_df.head())
