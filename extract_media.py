import json, csv
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

IN_JSONL = "test_7d.jsonl" #input
OUT_IMG_URLS = "image_urls.txt"
OUT_VID_URLS = "video_urls.txt"
OUT_INDEX = "images_index.csv"

def normalize_pbs_to_orig(url: str) -> str:
    # Turn https://pbs.twimg.com/media/XXX.jpg into https://pbs.twimg.com/media/XXX.jpg?name=orig
    # If ?format=... exists, keep it and set name=orig.
    try:
        u = urlparse(url)
        if "pbs.twimg.com" not in u.netloc:
            return url
        q = parse_qs(u.query)
        q["name"] = ["orig"]
        new_query = urlencode({k: v[0] for k, v in q.items()})
        return urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, u.fragment))
    except Exception:
        return url

def best_mp4_variant(video_info: dict) -> str | None:
    if not video_info:
        return None
    variants = video_info.get("variants", []) or []
    mp4s = [v for v in variants if v.get("content_type") == "video/mp4" and "url" in v]
    if not mp4s:
        return None
    mp4s.sort(key=lambda v: v.get("bitrate", -1), reverse=True)
    return mp4s[0]["url"]

img_urls = []
vid_urls = []
index_rows = []

with open(IN_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        kol = obj.get("kol_username", "")
        t = obj.get("tweet", {}) or {}
        tweet_id = t.get("id", "")
        created_at = t.get("createdAt", "")
        tweet_url = t.get("url") or t.get("twitterUrl") or ""

        media_list = ((t.get("extendedEntities") or {}).get("media")) or []
        for m in media_list:
            mtype = m.get("type", "")
            if mtype == "photo":
                u = m.get("media_url_https")
                if u:
                    u = normalize_pbs_to_orig(u)
                    img_urls.append(u)
                    index_rows.append([kol, tweet_id, created_at, tweet_url, "photo", u, ""]) # this keeps refernce to original tweet
            elif mtype == "video":
                # thumbnail image (optional)
                thumb = m.get("media_url_https")
                if thumb:
                    thumb = normalize_pbs_to_orig(thumb)
                    img_urls.append(thumb)
                    index_rows.append([kol, tweet_id, created_at, tweet_url, "video_thumb", thumb, ""])
                # actual mp4 (optional)
                mp4 = best_mp4_variant(m.get("video_info") or {})
                if mp4:
                    vid_urls.append(mp4)

# de-dupe while preserving order
def dedupe(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

img_urls = dedupe(img_urls)
vid_urls = dedupe(vid_urls)

with open(OUT_IMG_URLS, "w", encoding="utf-8") as f:
    f.write("\n".join(img_urls))

with open(OUT_VID_URLS, "w", encoding="utf-8") as f:
    f.write("\n".join(vid_urls))

with open(OUT_INDEX, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["kol_username","tweet_id","createdAt","tweet_url","media_kind","media_url","local_path"])
    w.writerows(index_rows)

print("images:", len(img_urls), "videos(mp4):", len(vid_urls), "index rows:", len(index_rows))
print("wrote:", OUT_IMG_URLS, OUT_VID_URLS, OUT_INDEX)
