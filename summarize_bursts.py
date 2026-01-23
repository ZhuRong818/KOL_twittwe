import json
import re
import pandas as pd
from collections import Counter
from urllib.parse import urlparse

# ========= 配置 =========
JSONL_FILE = "KOL.jsonl"
BURST_FILE = "burst_periods.csv"
OUT_FILE = "burst_event_summaries.csv"

TOP_K_KEYWORDS = 12
MIN_WORD_LEN = 3

# 简单停用词（可再加）
STOPWORDS = {
    "the","and","for","with","that","this","you","your","are","was","were","its","it's",
    "to","of","in","on","at","as","by","from","or","an","a","is","it","be","been",
    "i","we","they","he","she","them","his","her","our","us","my","me",
    "today","live","now","new","just","here","more"
}

WORD_RE = re.compile(r"[A-Za-z]{3,}")  # 只抓英文单词(>=3)


def parse_created_at(s: str):
    # "Tue Oct 07 12:38:00 +0000 2025"
    return pd.to_datetime(s, utc=True)


def get_text(tw: dict) -> str:
    # 兼容不同字段
    for k in ("text", "fullText", "content"):
        v = tw.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def engagement(tw: dict) -> int:
    likes = tw.get("likeCount") or tw.get("favoriteCount") or 0
    rts = tw.get("retweetCount") or 0
    replies = tw.get("replyCount") or 0
    quotes = tw.get("quoteCount") or 0
    try:
        return int(likes) + int(rts) + int(replies) + int(quotes)
    except Exception:
        return 0


def view_count(tw: dict) -> int:
    vc = tw.get("viewCount") or 0
    try:
        return int(vc)
    except Exception:
        return 0


def extract_urls(tweet_obj: dict):
    """
    从 tweet.entities.urls 和 text 里找链接。
    优先 expanded_url；否则 url。
    """
    urls = []

    ent = tweet_obj.get("entities") or {}
    url_list = ent.get("urls") or []
    for u in url_list:
        if isinstance(u, dict):
            ex = u.get("expanded_url") or u.get("expandedUrl")
            raw = u.get("url")
            if ex:
                urls.append(ex)
            elif raw:
                urls.append(raw)

    # 兜底：从文本抓 http(s)://
    text = get_text(tweet_obj)
    for m in re.findall(r"https?://\S+", text):
        urls.append(m.strip(")];.,\"'"))

    return urls


def normalize_link(u: str) -> str:
    """
    事件解释中，我们通常关心“同一个目标链接/域名”。
    对 t.co 来说 expanded_url 才有意义。
    这里做一个简单归一：只保留域名 + path 前 1-2 段。
    """
    try:
        p = urlparse(u)
        host = (p.netloc or "").lower()
        path = p.path or ""
        parts = [x for x in path.split("/") if x]
        short_path = "/".join(parts[:2])
        return f"{host}/{short_path}" if short_path else host
    except Exception:
        return u


def keywords_from_texts(texts):
    words = []
    for t in texts:
        for w in WORD_RE.findall((t or "").lower()):
            if len(w) < MIN_WORD_LEN:
                continue
            if w in STOPWORDS:
                continue
            words.append(w)
    return Counter(words)


# ---------- 1) 读入所有 tweets ----------
rows = []
with open(JSONL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        tw = obj["tweet"]
        rows.append({
            "kol": obj["kol_username"],
            "created_at": parse_created_at(tw["createdAt"]),
            "tweet_id": tw.get("id"),
            "text": get_text(tw),
            "url": tw.get("url") or tw.get("twitterUrl"),
            "viewCount": view_count(tw),
            "engagement": engagement(tw),
            "raw_tweet": tw,
        })

df = pd.DataFrame(rows)
df = df.sort_values("created_at")


# ---------- 2) 读入 burst periods ----------
burst_df = pd.read_csv(BURST_FILE)
burst_df["start"] = pd.to_datetime(burst_df["start"], utc=True)
burst_df["end"] = pd.to_datetime(burst_df["end"], utc=True)


# ---------- 3) 对每个 burst 生成事件解释 ----------
out = []

for _, b in burst_df.iterrows():
    kol = b["kol"]
    start = b["start"]
    end = b["end"]

    # 注意：你的 burst 是按小时桶生成的，这里 end+1小时更合理
    # 让窗口覆盖到最后一个小时结束
    end_inclusive = end + pd.Timedelta(hours=1)

    sub = df[(df["kol"] == kol) & (df["created_at"] >= start) & (df["created_at"] < end_inclusive)].copy()

    if sub.empty:
        out.append({
            "kol": kol,
            "start": start,
            "end": end,
            "tweet_count": 0,
            "top_metric": "",
            "top_value": 0,
            "top_tweet_url": "",
            "top_tweet_text": "",
            "keywords": "",
            "repeated_links": "",
            "event_hint": "no tweets found in window"
        })
        continue

    # top tweet：优先 viewCount，若全是 0 则用 engagement
    if sub["viewCount"].max() > 0:
        sub = sub.sort_values(["viewCount", "engagement"], ascending=False)
        top_metric = "viewCount"
        top_value = int(sub.iloc[0]["viewCount"])
    else:
        sub = sub.sort_values(["engagement"], ascending=False)
        top_metric = "engagement"
        top_value = int(sub.iloc[0]["engagement"])

    top = sub.iloc[0]

    # 关键词
    kw = keywords_from_texts(sub["text"].tolist())
    top_kws = [w for w, c in kw.most_common(TOP_K_KEYWORDS)]
    kw_str = ", ".join(top_kws)

    # 链接重复
    links = []
    for tw in sub["raw_tweet"].tolist():
        for u in extract_urls(tw):
            links.append(normalize_link(u))

    link_counts = Counter(links)
    repeated = [f"{u}({c})" for u, c in link_counts.most_common(10) if c >= 2]
    repeated_str = "; ".join(repeated)

    # 一个很粗的“事件提示”规则：有重复链接/提到直播/发布/会议/数据
    text_all = " ".join(sub["text"].fillna("").tolist()).lower()
    hint_tokens = ["live", "webinar", "masterclass", "earnings", "cpi", "fomc", "fed", "breaking", "release", "launched", "report"]
    is_event_like = (len(repeated) > 0) or any(t in text_all for t in hint_tokens)

    out.append({
        "kol": kol,
        "start": start,
        "end": end,
        "tweet_count": int(len(sub)),
        "top_metric": top_metric,
        "top_value": top_value,
        "top_tweet_url": top["url"],
        "top_tweet_text": (top["text"] or "")[:280],
        "keywords": kw_str,
        "repeated_links": repeated_str,
        "event_hint": "likely event/announcement" if is_event_like else "burst but unclear topic"
    })

out_df = pd.DataFrame(out)
out_df.to_csv(OUT_FILE, index=False, encoding="utf-8")

print(f"✅ Wrote {OUT_FILE} with {len(out_df)} burst summaries.")
print(out_df.head(10))
