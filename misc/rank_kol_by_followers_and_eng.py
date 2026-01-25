import os
import time
import math
import random
from datetime import datetime, timedelta, timezone

import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from dateutil import parser as dtparser

# =========================
# CONFIG
# =========================
API_KEY = os.getenv("TWITTERAPI_IO_KEY")
if not API_KEY:
    raise SystemExit("Missing TWITTERAPI_IO_KEY")

PROFILE_URL = "https://api.twitterapi.io/twitter/user/info"
TIMELINE_URL = "https://api.twitterapi.io/twitter/user/last_tweets"

HEADERS = {"X-API-Key": API_KEY}

MIN_FOLLOWERS = 400_000

RECENT_TWEETS = 20          # 拉最近多少条（省钱）
LOOKBACK_DAYS = 50          # 只算最近 50 天
MIN_TWEETS = 3              # 至少多少条有效推
TOP_K = 200                 # 最终输出多少人

OUT_FILE = "kol_rank_followers_engagement.csv"
TOP200_TXT = "top200_handles.txt"   # 新增：只包含前200 handle 的文件

INPUT_FILE = "misc_candidates.txt"


# =========================
# helpers
# =========================
def sleep_a_bit():
    time.sleep(0.3 + random.random() * 0.3)


def is_retweet(t: dict) -> bool:
    text = (t.get("text") or "")
    # 有些 API 可能给 type=retweet
    return text.startswith("RT @") or (t.get("type") == "retweet")


def engagement(t: dict) -> int:
    return int(
        (t.get("likeCount", 0) or 0)
        + (t.get("replyCount", 0) or 0)
        + (t.get("retweetCount", 0) or 0)
        + (t.get("quoteCount", 0) or 0)
    )


def request_json(url, params, retries=6):
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=60)

            if r.status_code == 429:
                time.sleep(5 + i * 2 + random.random())
                continue

            if 500 <= r.status_code < 600:
                time.sleep(min(2 ** i, 30) + random.random())
                continue

            r.raise_for_status()
            return r.json()

        except Exception as e:
            last_err = e
            time.sleep(min(2 ** i, 30) + random.random())

    raise RuntimeError(f"Request failed: {url} | last_error={last_err}")


def pick_tweets(data: dict):
    # 兼容不同返回结构
    tweets = data.get("tweets") or data.get("data") or data.get("items") or data.get("results") or []
    return tweets if isinstance(tweets, list) else []


# =========================
# core logic
# =========================
def score_user(username: str):
    username = username.strip().lstrip("@")
    if not username:
        return None

    # -------- profile --------
    prof = request_json(PROFILE_URL, {"userName": username})
    user = prof.get("user") or prof.get("data") or prof
    if isinstance(prof.get("data"), dict) and isinstance(prof.get("data").get("data"), dict):
        user = prof["data"]["data"]

    followers = user.get("followersCount") or user.get("followers_count") or user.get("followers") or 0
    followers = int(followers or 0)

    # ✅ 排除 follower < 400k
    if followers < 400_000:
        return None

    # -------- timeline --------
    data = request_json(
        TIMELINE_URL,
        {"userName": username, "includeReplies": "false", "cursor": ""}
    )
    tweets = pick_tweets(data)

    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)

    eng_list = []
    for t in tweets:
        if is_retweet(t):
            continue

        created = t.get("createdAt") or t.get("created_at")
        if not created:
            continue

        try:
            dt = dtparser.parse(created)
        except Exception:
            continue

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt < cutoff:
            continue

        eng_list.append(engagement(t))

    if len(eng_list) < MIN_TWEETS:
        return None

    median_eng = float(np.median(eng_list))

    # -------- SCORE --------
    # 只看 follower + 最近 engagement（各 0.5）
    score = (
        0.5 * math.log1p(followers) +
        0.5 * math.log1p(median_eng)
    )

    return {
        "username": username,
        "followers": followers,
        "median_engagement": median_eng,
        "score": score,
        "tweets_used": len(eng_list)
    }


# =========================
# main
# =========================
def main():
    if not os.path.exists(INPUT_FILE):
        raise SystemExit(f"Missing input file: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        users = [line.strip().lstrip("@") for line in f if line.strip()]

    # 去重保持顺序
    seen = set()
    unique_users = []
    for u in users:
        if u and u not in seen:
            seen.add(u)
            unique_users.append(u)

    rows = []

    for u in tqdm(unique_users, desc="Ranking KOLs"):
        try:
            r = score_user(u)
            if r:
                rows.append(r)
        except Exception as e:
            print("skip", u, e)

        sleep_a_bit()

    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid users found.")
        return

    df = df.sort_values("score", ascending=False).head(TOP_K).reset_index(drop=True)

    # 1) 保存 CSV
    df.to_csv(OUT_FILE, index=False)
    print(f"\nSaved: {OUT_FILE}")
    print(df.head(20).to_string(index=False))

    # 2) 保存 Top200 handle txt（只写 username，一行一个）
    with open(TOP200_TXT, "w", encoding="utf-8") as f:
        for h in df["username"].head(200):
            f.write(f"{h}\n")

    print(f"Saved: {TOP200_TXT}")


if __name__ == "__main__":
    main()
