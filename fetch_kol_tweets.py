#!/usr/bin/env python3
"""
fetch_kol_tweets.py

Goal: Given a username list (one handle per line), fetch ORIGINAL English tweets
from each account over a time window (past N years OR past N days), and save as JSONL.

- Uses twitterapi.io:
  - /twitter/tweet/advanced_search

- Free-tier note:
  - You said you hit: "one request every 5 seconds" -> we enforce that by default.

Output format (JSONL, 1 tweet per line):
{"kol_username": "...", "fetched_at_utc": "...", "tweet": {...raw tweet obj...}}
"""

import os
import json
import time
import argparse
import datetime as dt
from dateutil.relativedelta import relativedelta
import requests

BASE_URL = "https://api.twitterapi.io"

# Default: one request per ~5 seconds (safe for free tier)
DEFAULT_MIN_REQUEST_INTERVAL = 5.02


# -----------------------------
# Rate limiting + request helper
# -----------------------------
_last_request_time = 0.0

def rate_limit(min_interval_s: float):
    """Global simple rate limiter: ensures at least min_interval_s between requests."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < min_interval_s:
        time.sleep(min_interval_s - elapsed)
    _last_request_time = time.time()


def request_json(api_key: str, endpoint: str, params: dict, min_interval_s: float, timeout: int = 30) -> dict:
    """GET request with rate limit + basic 429 backoff."""
    rate_limit(min_interval_s)

    url = f"{BASE_URL}{endpoint}"
    headers = {"X-API-Key": api_key}

    r = requests.get(url, headers=headers, params=params or {}, timeout=timeout)

    # If still hit 429, back off harder and retry once
    if r.status_code == 429:
        # Many APIs include retry hints, but we keep it simple:
        backoff = max(10.0, min_interval_s * 2)
        print(f"[WARN] HTTP 429 Too Many Requests. Sleeping {backoff:.1f}s then retrying once...")
        time.sleep(backoff)
        rate_limit(min_interval_s)
        r = requests.get(url, headers=headers, params=params or {}, timeout=timeout)

    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")

    return r.json()


# -----------------------------
# Utilities
# -----------------------------
def utc_str(d: dt.datetime) -> str:
    # Keep your previous format; if your API returns empty results,
    # change to YYYY-MM-DD (see note in chat)
    return d.strftime("%Y-%m-%d_%H:%M:%S_UTC")


def read_usernames(path: str) -> list[str]:
    names: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            u = line.strip().lstrip("@")
            if u and not u.startswith("#"):
                names.append(u)

    # de-dup preserving order
    seen = set()
    out: list[str] = []
    for u in names:
        k = u.lower()
        if k not in seen:
            seen.add(k)
            out.append(u)
    return out


# -----------------------------
# twitterapi.io wrapper
# -----------------------------
def advanced_search(api_key: str, query: str, cursor: str, query_type: str, min_interval_s: float) -> dict:
    return request_json(
        api_key=api_key,
        endpoint="/twitter/tweet/advanced_search",
        params={"query": query, "queryType": query_type, "cursor": cursor},
        min_interval_s=min_interval_s,
    )


# -----------------------------
# Core fetcher
# -----------------------------
def fetch_user_tweets_advanced_search(
    api_key: str,
    username: str,
    since_utc: str,
    until_utc: str,
    english_only: bool,
    originals_only: bool,
    query_type: str,
    min_interval_s: float,
    max_pages: int = 10_0,
) -> list[dict]:
    """
    Fetch tweets for a user via advanced_search.
    - english_only: adds "lang:en"
    - originals_only: adds "-filter:retweets -filter:replies"
    """
    lang_clause = " lang:en" if english_only else ""
    filter_clause = " -filter:retweets -filter:replies" if originals_only else ""

    query = f"from:{username}{lang_clause}{filter_clause} since:{since_utc} until:{until_utc}"

    cursor = ""
    all_tweets: list[dict] = []
    seen_ids = set()
    empty_pages = 0

    for _page in range(max_pages):
        data = advanced_search(api_key, query, cursor=cursor, query_type=query_type, min_interval_s=min_interval_s)
        tweets = data.get("tweets", []) or []

        new_count = 0
        for t in tweets:
            tid = str(t.get("id", ""))
            if tid and tid not in seen_ids:
                seen_ids.add(tid)
                all_tweets.append(t)
                new_count += 1

        if new_count == 0:
            empty_pages += 1
        else:
            empty_pages = 0

        # stop conditions
        has_next = bool(data.get("has_next_page"))
        next_cursor = data.get("next_cursor", "") or ""

        if not has_next or not next_cursor:
            break

        # defensive stop if API keeps returning empty pages
        if empty_pages >= 2:
            break

        cursor = next_cursor

    return all_tweets


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--api_key", default=os.getenv("TWITTERAPI_IO_KEY"))
    ap.add_argument("--usernames", required=True, help="Path to username list file (one per line).")
    ap.add_argument("--out", default="kol_tweets.jsonl", help="Output JSONL path")

    # Time window
    ap.add_argument("--years", type=int, default=2, help="Fetch tweets from past N years (ignored if --days>0)")
    ap.add_argument("--days", type=int, default=0, help="If >0, fetch tweets from last N days instead of years")

    # Content options
    ap.add_argument("--english_only", action="store_true", help="If set, only fetch English tweets (lang:en)")
    ap.add_argument("--originals_only", action="store_true", help="If set, exclude retweets and replies")
    ap.add_argument("--query_type", default="Latest", choices=["Latest", "Top"])

    # Rate limit controls
    ap.add_argument("--min_interval", type=float, default=DEFAULT_MIN_REQUEST_INTERVAL,
                    help="Min seconds between API requests (free-tier often needs ~5s)")

    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Provide --api_key or set TWITTERAPI_IO_KEY.")

    usernames = read_usernames(args.usernames)
    if not usernames:
        raise SystemExit("No usernames found in file.")

    now = dt.datetime.utcnow().replace(microsecond=0)
    if args.days and args.days > 0:
        since = now - dt.timedelta(days=args.days)
    else:
        since = now - relativedelta(years=args.years)

    since_s = utc_str(since)
    until_s = utc_str(now)

    fetched_at = utc_str(now)

    total_written = 0
    with open(args.out, "w", encoding="utf-8") as f_out:
        for i, u in enumerate(usernames, start=1):
            try:
                tweets = fetch_user_tweets_advanced_search(
                    api_key=args.api_key,
                    username=u,
                    since_utc=since_s,
                    until_utc=until_s,
                    english_only=args.english_only,
                    originals_only=args.originals_only,
                    query_type=args.query_type,
                    min_interval_s=args.min_interval,
                )

                for t in tweets:
                    f_out.write(json.dumps(
                        {"kol_username": u, "fetched_at_utc": fetched_at, "tweet": t},
                        ensure_ascii=False
                    ) + "\n")

                total_written += len(tweets)
                print(f"[{i}/{len(usernames)}] @{u}: wrote {len(tweets)} tweets")

            except Exception as e:
                print(f"[{i}/{len(usernames)}] @{u}: ERROR: {e}")

    print(f"Done. Wrote {total_written} tweets to {args.out}")


if __name__ == "__main__":
    main()
