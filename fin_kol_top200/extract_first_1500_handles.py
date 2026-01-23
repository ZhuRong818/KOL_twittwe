import json

INPUT_JSON = "fin_KOL_raw.json"
OUTPUT_TXT = "handles_1500.txt"
N = 1500

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

handles = []
for item in data:
    name = item.get("screen_name")
    if not name:
        continue
    if name.lower() == "mock data":
        continue
    handles.append(name)
    if len(handles) >= N:
        break

with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    for h in handles:
        f.write(h + "\n")

print(f"Saved {len(handles)} handles to {OUTPUT_TXT}")
