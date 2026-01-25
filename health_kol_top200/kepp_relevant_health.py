import json

with open("dataset_twitter-kol-discovery-find-influencers-fast-0-5-1k-2025_2026-01-24_13-32-16-438.json") as f:
    data = json.load(f)

def count_value(obj: dict) -> int:
    val = obj.get("count", 0)
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


# 按 count 从高到低排序
sorted_users = sorted(
    data,
    key=count_value,
    reverse=True
)

# 取前 100 个 handle
top_100 = [u["screen_name"] for u in sorted_users[:400]]

# 保存
with open("top_100_handles.txt", "w") as f:
    for h in top_100:
        f.write(h + "\n")

print(len(top_100), "handles saved")
