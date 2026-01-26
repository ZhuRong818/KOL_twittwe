import os, csv, hashlib, time
import requests
from urllib.parse import urlparse, parse_qs

IN_INDEX = "images_index.csv"
OUT_INDEX = "images_index_filled.csv"
OUT_DIR = "images"

os.makedirs(OUT_DIR, exist_ok=True)

def ext_from_url(url: str) -> str:
    # Twitter images often include ?format=jpg
    q = parse_qs(urlparse(url).query)
    fmt = (q.get("format", ["jpg"])[0]).lower()
    if fmt == "jpeg":
        fmt = "jpg"
    if fmt not in {"jpg", "png", "webp"}:
        fmt = "jpg"
    return fmt

def filename_from_bytes(content: bytes, url: str) -> str:
    h = hashlib.sha256(content).hexdigest()[:24]
    return f"{h}.{ext_from_url(url)}"

def download(url: str, timeout=25) -> tuple[str | None, int | None]:
    # Returns (local_path, bytes) or (None, None)
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            content = r.content
            if len(content) < 200:
                raise RuntimeError("Too small / not image")
            fn = filename_from_bytes(content, url)
            path = os.path.join(OUT_DIR, fn)
            if not os.path.exists(path):
                with open(path, "wb") as f:
                    f.write(content)
            return path, len(content)
        except Exception:
            time.sleep(0.6 * (2 ** attempt))
    return None, None

rows_out = []
ok = 0
total = 0

with open(IN_INDEX, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames[:]  # keep original columns
    if "local_path" not in fieldnames:
        fieldnames.append("local_path")

    for row in reader:
        total += 1
        url = row.get("media_url", "").strip()
        if not url:
            rows_out.append(row)
            continue

        # Skip if already filled and file exists
        existing = row.get("local_path", "").strip()
        if existing and os.path.exists(existing):
            ok += 1
            rows_out.append(row)
            continue

        path, nbytes = download(url)
        if path:
            row["local_path"] = path
            ok += 1
        else:
            row["local_path"] = ""  # failed
        rows_out.append(row)

# write updated index
with open(OUT_INDEX, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows_out)

print(f"Downloaded/linked {ok}/{total}")
print("Wrote:", OUT_INDEX)
