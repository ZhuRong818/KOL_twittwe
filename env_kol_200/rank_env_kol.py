#!/usr/bin/env python3
"""
Rank X (Twitter) KOLs by engagement using twitterapi.io

Input:
  - second_cannidatas.txt (one X handle per line, no @ needed)

Filters:
  - followers > 200,000
  - at least 1 non-retweet tweet in the last ACTIVE_WITHIN_DAYS
  - evaluate most recent 20 tweets

Outputs:
  - ranked_kols.csv
  - top300.txt

Auth:
  export TWITTERAPI_IO_KEY="YOUR_KEY"

Change vs your current algorithm:
  - Engagement score is now RECENCY-AWARE (tweet-level time decay + velocity),
    so older tweets don't keep dominating forever.
"""

import os
import re
import time
import math
import csv
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

# =========================
# CONFIG (edit if needed)
# =========================
API_KEY = os.getenv("TWITTERAPI_IO_KEY", "").strip()
if not API_KEY:
    raise SystemExit("Missing TWITTERAPI_IO_KEY. Run: export TWITTERAPI_IO_KEY='xxx'")

BASE = "https://api.twitterapi.io"
USER_INFO_URL = f"{BASE}/twitter/user/info"          # ?userName=
LAST_TWEETS_URL = f"{BASE}/twitter/user/last_tweets" # ?userName=&count=

INPUT_HANDLES_FILE = "top_300_handles.txt"

MIN_FOLLOWERS = 100_000
TWEETS_TO_EVAL = 20
ACTIVE_WITHIN_DAYS = 60

SLEEP_SEC = 0.25  # be polite / avoid rate-limit bursts

OUT_CSV = "ranked_kols.csv"
OUT_TOP_TXT = "top120.txt"
TOP_N = 120

HEADERS = {"X-API-Key": API_KEY}

# Engagement definition (per tweet, non-retweets only)
W_LIKE = 1.0
W_REPLY = 1.0
W_RETWEET = 1.0
W_QUOTE = 1.0

# Legacy mix knobs (still used, but now applied to recency-aware engagement_score)
W_ABS = 0.7
W_RATE = 0.3

# =========================
# NEW: Recency-aware scoring knobs
# =========================
# Exponential decay: weight halves every HALF_LIFE_DAYS
HALF_LIFE_DAYS = 7.0

# Velocity: engagement divided by (age_hours + TAU_HOURS)^ALPHA
TAU_HOURS = 12.0
ALPHA = 1.2

# Blend velocity + decayed engagement
W_VELOCITY = 0.7
W_DECAYED = 0.3


# =========================
# Helpers
# =========================
def clean_handle(line: str) -> Optional[str]:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    s = s.lstrip("@")
    s = re.sub(r"[^A-Za-z0-9_]", "", s)
    return s or None


def parse_created_at(tweet: Dict[str, Any]) -> Optional[datetime]:
    """
    twitterapi.io commonly returns createdAt like:
      "Tue Oct 07 12:38:00 +0000 2025"
    We'll parse it robustly.
    """
    v = tweet.get("createdAt") or tweet.get("created_at")
    if not v:
        return None
    for fmt in ("%a %b %d %H:%M:%S %z %Y", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(v, fmt)
        except ValueError:
            pass
    try:
        from dateutil import parser as dtparser  # type: ignore
        return dtparser.parse(v)
    except Exception:
        return None


def is_retweet(tweet: Dict[str, Any]) -> bool:
    ttype = (tweet.get("type") or "").lower()
    if "retweet" in ttype:
        return True

    for k in ("retweetedTweet", "retweeted_tweet", "retweeted_status"):
        if k in tweet and tweet.get(k):
            return True

    for k in ("isRetweet", "is_retweet", "retweeted"):
        if tweet.get(k) is True:
            return True

    txt = (tweet.get("text") or "").strip()
    if txt.startswith("RT @"):
        return True

    return False


def get_counts(tweet: Dict[str, Any]) -> Tuple[int, int, int, int]:
    rt = int(tweet.get("retweetCount") or tweet.get("retweet_count") or 0)
    rp = int(tweet.get("replyCount") or tweet.get("reply_count") or 0)
    lk = int(tweet.get("likeCount") or tweet.get("favoriteCount") or tweet.get("like_count") or 0)
    qt = int(tweet.get("quoteCount") or tweet.get("quote_count") or 0)
    return rt, rp, lk, qt


def safe_get_json(url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def fetch_user_info(username: str) -> Optional[Dict[str, Any]]:
    return safe_get_json(USER_INFO_URL, {"userName": username})


def fetch_last_tweets(username: str, count: int) -> Optional[List[Dict[str, Any]]]:
    data = safe_get_json(LAST_TWEETS_URL, {"userName": username, "count": count})
    if not data:
        return None

    if isinstance(data, list):
        return data
    if isinstance(data.get("data"), dict) and isinstance(data["data"].get("tweets"), list):
        return data["data"]["tweets"]
    for key in ("tweets", "data", "items", "results"):
        if key in data and isinstance(data[key], list):
            return data[key]

    return None


def compute_score(total_eng: float, followers: int) -> Tuple[float, float]:
    """
    Returns (engagement_rate, score)
    engagement_rate: total_eng / followers
    score: mix of absolute + rate, using logs for stability
    """
    if followers <= 0:
        return 0.0, 0.0
    rate = total_eng / followers
    score = W_ABS * math.log1p(total_eng) + W_RATE * math.log1p(rate * 1e6)
    return rate, score


# =========================
# NEW: Recency-aware scoring pieces
# =========================
def tweet_age_hours(tweet_dt: datetime, now: datetime) -> float:
    return max(0.0, (now - tweet_dt).total_seconds() / 3600.0)


def exp_decay_weight(age_days: float, half_life_days: float) -> float:
    return math.exp(-math.log(2) * age_days / half_life_days)


def engagement_of_tweet(t: Dict[str, Any]) -> float:
    rt, rp, lk, qt = get_counts(t)
    return (W_RETWEET * rt + W_REPLY * rp + W_LIKE * lk + W_QUOTE * qt)


# =========================
# Main
# =========================
def main():
    now = datetime.now(timezone.utc)
    active_cutoff = now - timedelta(days=ACTIVE_WITHIN_DAYS)

    # Read handles
    handles: List[str] = []
    with open(INPUT_HANDLES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            h = clean_handle(line)
            if h:
                handles.append(h)

    if not handles:
        raise SystemExit(f"No valid handles found in {INPUT_HANDLES_FILE}")

    rows: List[Dict[str, Any]] = []

    for i, h in enumerate(handles, 1):
        # --- user info ---
        info = fetch_user_info(h)
        time.sleep(SLEEP_SEC)

        if not info:
            continue

        followers = (
            info.get("followersCount")
            or info.get("followers_count")
            or info.get("followers")
            or (info.get("data", {}) if isinstance(info.get("data"), dict) else {}).get("followers")
            or (info.get("data", {}) if isinstance(info.get("data"), dict) else {}).get("followersCount")
            or 0
        )
        try:
            followers = int(followers)
        except Exception:
            followers = 0

        if followers <= MIN_FOLLOWERS:
            continue

        # --- last tweets ---
        tweets = fetch_last_tweets(h, TWEETS_TO_EVAL)
        time.sleep(SLEEP_SEC)

        if not tweets:
            continue

        # Only non-retweets
        non_rt = [t for t in tweets if isinstance(t, dict) and not is_retweet(t)]

        # Must have at least 1 non-retweet in the last ACTIVE_WITHIN_DAYS
        dated_non_rt: List[Tuple[Dict[str, Any], datetime]] = []
        for t in non_rt:
            dt = parse_created_at(t)
            if dt:
                dated_non_rt.append((t, dt))

        if not dated_non_rt:
            continue

        most_recent_non_rt = max(dt for _, dt in dated_non_rt)
        if most_recent_non_rt < active_cutoff:
            continue

        # Engagement stats (raw) + Recency-aware scores
        total_eng = 0.0  # raw total (for reporting)
        total_velocity = 0.0
        total_decayed = 0.0

        total_rt = total_rp = total_lk = total_qt = 0

        for t, dt in dated_non_rt:
            rt, rp, lk, qt = get_counts(t)
            total_rt += rt
            total_rp += rp
            total_lk += lk
            total_qt += qt

            eng = (W_RETWEET * rt + W_REPLY * rp + W_LIKE * lk + W_QUOTE * qt)
            total_eng += eng

            age_h = tweet_age_hours(dt, now)
            age_d = age_h / 24.0

            # Velocity (age-normalized)
            vel = eng / ((age_h + TAU_HOURS) ** ALPHA)
            total_velocity += vel

            # Exponential decay
            w = exp_decay_weight(age_d, HALF_LIFE_DAYS)
            total_decayed += eng * w

        n_eval = len(dated_non_rt)
        avg_eng = total_eng / n_eval if n_eval else 0.0

        # Final recency-aware engagement score used for ranking
        engagement_score = W_VELOCITY * total_velocity + W_DECAYED * total_decayed

        # Keep your old compute_score structure, but feed engagement_score (not raw total)
        engagement_rate, score = compute_score(engagement_score, followers)

        rows.append(
            {
                "handle": h,
                "followers": followers,
                "non_rt_tweets_in_20": n_eval,
                "most_recent_non_rt_utc": most_recent_non_rt.isoformat(),

                # Raw totals (helpful to inspect)
                "total_like": total_lk,
                "total_reply": total_rp,
                "total_retweet": total_rt,
                "total_quote": total_qt,
                "total_engagement_raw": round(total_eng, 2),
                "avg_engagement_raw": round(avg_eng, 2),

                # New scoring components
                "velocity_score": round(total_velocity, 6),
                "decayed_engagement": round(total_decayed, 2),
                "engagement_score": round(engagement_score, 6),

                # Ranking score
                "engagement_rate": round(engagement_rate, 8),
                "score": round(score, 6),
            }
        )

        if i % 50 == 0:
            print(f"Processed {i}/{len(handles)} handles... kept {len(rows)}")

    # Rank
    rows.sort(key=lambda r: r["score"], reverse=True)
    for idx, r in enumerate(rows, 1):
        r["rank"] = idx

    # Write CSV
    fieldnames = [
        "rank",
        "handle",
        "followers",
        "non_rt_tweets_in_20",
        "most_recent_non_rt_utc",
        "total_like",
        "total_reply",
        "total_retweet",
        "total_quote",
        "total_engagement_raw",
        "avg_engagement_raw",
        "velocity_score",
        "decayed_engagement",
        "engagement_score",
        "engagement_rate",
        "score",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Write top N TXT
    top = rows[:TOP_N]
    with open(OUT_TOP_TXT, "w", encoding="utf-8") as f:
        for r in top:
            f.write(r["handle"] + "\n")

    print("\nDone.")
    print(f"Kept {len(rows)} accounts meeting filters.")
    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_TOP_TXT}")


if __name__ == "__main__":
    main()
