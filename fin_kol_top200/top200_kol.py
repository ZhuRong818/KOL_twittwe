import os
import time
import requests
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from dateutil import parser
from tqdm import tqdm

# =========================
# API key
# =========================
API_KEY = os.getenv("TWITTERAPI_IO_KEY")
if not API_KEY:
    raise SystemExit("Missing API key. Set TWITTERAPI_IO_KEY in your environment.")

BASE_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"

# twitterapi.io advanced_search commonly uses X-API-Key
HEADERS = {
    "X-API-Key": API_KEY,
    "User-Agent": "fin-kol-top200/1.0"
}

# =========================
# COST CONTROL SETTINGS
# =========================
MAX_TWEETS_PER_QUERY = 200   # per query cap
LANG = "en"
REQUEST_TIMEOUT_S = 150
MAX_RETRIES = 3
BACKOFF_BASE_S = 5.0

# Past 2 months
END_TIME = datetime.now(timezone.utc)
START_TIME = END_TIME - timedelta(days=60)

# =========================
# QUERY SET
# =========================
QUERIES = [
    "CPI", "inflation", "FOMC", "Fed meeting", "rate cut", "rate decision",
    "yield curve", "Treasury", "real yields",
    "$SPY", "$QQQ", "$AAPL", "$NVDA", "earnings", "earnings call", "guidance",
    "buyback", "ETF inflows",
    "AI capex", "semiconductor cycle", "China economy", "commercial real estate",
    "bank stress", "liquidity conditions",
    "options flow", "gamma", "0DTE", "technical setup", "support resistance", "breakout",
    "$BTC", "Bitcoin ETF"
]

# =========================
# Helpers
# =========================
def is_retweet(tweet: dict) -> bool:
    # Based on your debug: retweeted_tweet exists
    text = (tweet.get("text") or "").strip()
    if text.startswith("RT @"):
        return True
    if tweet.get("type") == "retweet":
        return True
    if tweet.get("retweeted_tweet") is not None:
        return True
    return False

def utc_str(d: datetime) -> str:
    # Most compatible in query syntax: use date only
    return d.strftime("%Y-%m-%d")

def get_tweets(data: dict) -> list:
    tweets = data.get("tweets") or data.get("data") or data.get("results") or data.get("items") or []
    return tweets if isinstance(tweets, list) else []

def get_next_cursor(data: dict) -> str:
    cur = data.get("next_cursor") or data.get("nextCursor") or data.get("cursor") or ""
    return cur if isinstance(cur, str) else ""

def has_next_page(data: dict) -> bool:
    return bool(data.get("has_next_page") or data.get("hasNextPage"))

def extract_username(tweet: dict) -> str | None:
    # Your debug shows "author" exists
    author = tweet.get("author") or {}
    username = author.get("username") or author.get("screen_name") or author.get("userName")
    if not username:
        return None
    return str(username).lstrip("@")

def extract_created_at(tweet: dict) -> str | None:
    return tweet.get("createdAt") or tweet.get("created_at")

# =========================
# Collect author stats
# =========================
authors = defaultdict(lambda: {
    "views": [],
    "eng": [],
    "days": set(),
    "n_posts": 0,
    "original_posts": 0
})

print(f"Time window (UTC): {START_TIME.date()} to {END_TIME.date()}")
print(f"Queries: {len(QUERIES)} | Max tweets/query: {MAX_TWEETS_PER_QUERY}")

since_s = utc_str(START_TIME)
until_s = utc_str(END_TIME)

for query in tqdm(QUERIES, desc="Searching queries"):
    query_str = f"{query} lang:{LANG} since:{since_s} until:{until_s}"

    collected = 0
    cursor = ""  # start

    while collected < MAX_TWEETS_PER_QUERY:
        params = {
            "query": query_str,
            "queryType": "Latest",
            "cursor": cursor
        }

        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=REQUEST_TIMEOUT_S)
                r.raise_for_status()
                data = r.json()
                last_err = None
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_err = e
                if attempt >= MAX_RETRIES:
                    break
                backoff = BACKOFF_BASE_S * (2 ** (attempt - 1))
                print(f"[WARN] Timeout/connection error. Retry {attempt}/{MAX_RETRIES} after {backoff:.1f}s...")
                time.sleep(backoff)

        if last_err is not None:
            raise last_err

        tweets = get_tweets(data)
        if not tweets:
            break

        for t in tweets:
            if collected >= MAX_TWEETS_PER_QUERY:
                break

            username = extract_username(t)
            if not username:
                continue

            created_at = extract_created_at(t)
            if not created_at:
                continue
            try:
                day = parser.parse(created_at).date()
            except Exception:
                continue

            views = t.get("viewCount", 0) or 0
            like = t.get("likeCount", 0) or 0
            reply = t.get("replyCount", 0) or 0
            rt = t.get("retweetCount", 0) or 0
            quote = t.get("quoteCount", 0) or 0
            eng = like + reply + rt + quote

            a = authors[username]
            a["views"].append(int(views))
            a["eng"].append(int(eng))
            a["days"].add(day)
            a["n_posts"] += 1
            if not is_retweet(t):
                a["original_posts"] += 1

            collected += 1

        # pagination
        if not has_next_page(data):
            break
        cursor = get_next_cursor(data)
        if not cursor:
            break

# =========================
# Filter + score
# =========================
records = []

for username, a in authors.items():
    n = a["n_posts"]
    active_days = len(a["days"])
    if n < 3:
        continue
    if active_days < 3:
        continue

    originality = a["original_posts"] / n if n else 0.0
    if originality < 0.5:
        continue

    median_views = float(np.median(a["views"])) if a["views"] else 0.0
    median_eng = float(np.median(a["eng"])) if a["eng"] else 0.0

    # IMPORTANT: if viewCount missing/zero, fall back to engagement threshold
    if median_views <= 1:
        if median_eng < 5:
            continue
    else:
        if median_views < 300:
            continue

    records.append({
        "username": username,
        "median_views": median_views,
        "median_engagement": median_eng,
        "active_days": active_days,
        "n_posts_sampled": n,
        "originality_ratio": originality
    })

df = pd.DataFrame(records)
if df.empty:
    print("No candidates found.")
    print("DEBUG authors_total:", len(authors))
    # Print a few authors to inspect quickly
    sample = list(authors.items())[:3]
    print("DEBUG sample_authors:", [(u, v["n_posts"], len(v["days"]), float(np.median(v["views"])) if v["views"] else 0.0) for u, v in sample])
    raise SystemExit(1)

# Tested hybrid influence score
df["score"] = (
    0.6 * np.log1p(df["median_views"])
    + 0.3 * np.log1p(df["median_engagement"])
    + 0.1 * np.log1p(df["active_days"])
)

top200 = (df.sort_values("score", ascending=False)
            .head(200)
            .reset_index(drop=True))

top200.to_csv("top200_financial_kols.csv", index=False)

print("\nSaved output: top200_financial_kols.csv")
print("\nTop 10 preview:")
print(top200.head(10).to_string(index=False))
print(f"\nCandidates after filtering: {len(df)} | Total authors seen: {len(authors)}")
