#!/usr/bin/env python3
"""
fetch_kol_tweets.py  (TIMELINE / PROFILE FEED VERSION)

✅ Reliable for: original + retweet/repost + quote + reply
✅ Extracts image/video links (also from quoted/retweeted embedded tweets)
✅ Uses timeline endpoint: /twitter/user/last_tweets (mirrors profile feed)

IMPORTANT: Your twitterapi.io response shape is:
  { status, code, msg, data, has_next_page, next_cursor }

…and the tweets are inside:  data["tweets"]  (NOT top-level "tweets")

Usage examples:
  python fetch_kol_tweets.py --usernames handles.txt --out out.jsonl --days 7
  python fetch_kol_tweets.py --usernames handles.txt --out out.jsonl --days 30 --exclude_replies
  python fetch_kol_tweets.py --usernames handles.txt --out out.jsonl --days 10 --retweets_only
  python fetch_kol_tweets.py --usernames handles.txt --out out.jsonl --debug
"""

import os
import json
import time
import argparse
import datetime as dt
from typing import Any, Dict, List, Optional

import requests
from dateutil import parser as dtparser


BASE_URL = "https://api.twitterapi.io"
TIMELINE_ENDPOINT = "/twitter/user/last_tweets"

DEFAULT_MIN_REQUEST_INTERVAL = 5.02
DEFAULT_TIMEOUT = 60

_last_request_time = 0.0


# -----------------------------
# Rate limiting + request helper
# -----------------------------
def rate_limit(min_interval_s: float):
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < min_interval_s:
        time.sleep(min_interval_s - elapsed)
    _last_request_time = time.time()


def request_json(
    api_key: str,
    endpoint: str,
    params: Dict[str, Any],
    min_interval_s: float,
    timeout: int,
    max_retries: int = 4,
) -> Dict[str, Any]:
    url = f"{BASE_URL}{endpoint}"
    headers = {"X-API-Key": api_key}

    for attempt in range(1, max_retries + 1):
        rate_limit(min_interval_s)
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)

            if r.status_code == 429:
                backoff = max(10.0, min_interval_s * 2) * attempt
                print(f"[WARN] 429 rate limited. Sleep {backoff:.1f}s then retry ({attempt}/{max_retries})")
                time.sleep(backoff)
                continue

            if r.status_code != 200:
                body = (r.text or "")[:400]
                if r.status_code in (500, 502, 503, 504) and attempt < max_retries:
                    backoff = 3.0 * attempt
                    print(f"[WARN] HTTP {r.status_code}. Sleep {backoff:.1f}s then retry ({attempt}/{max_retries})")
                    time.sleep(backoff)
                    continue
                raise RuntimeError(f"HTTP {r.status_code}: {body}")

            return r.json()

        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            if attempt >= max_retries:
                raise RuntimeError(f"Network error after retries: {e}") from e
            backoff = 5.0 * attempt
            print(f"[WARN] {type(e).__name__}. Sleep {backoff:.1f}s then retry ({attempt}/{max_retries})")
            time.sleep(backoff)

    raise RuntimeError("request_json failed unexpectedly")


# -----------------------------
# Helpers
# -----------------------------
def read_usernames(path: str) -> List[str]:
    names: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            u = line.strip().lstrip("@")
            if u and not u.startswith("#"):
                names.append(u)

    seen = set()
    out: List[str] = []
    for u in names:
        k = u.lower()
        if k not in seen:
            seen.add(k)
            out.append(u)
    return out


def parse_created_at(tweet: Dict[str, Any]) -> Optional[dt.datetime]:
    s = tweet.get("createdAt") or tweet.get("created_at")
    if not s:
        return None
    try:
        return dtparser.parse(s).astimezone(dt.timezone.utc)
    except Exception:
        return None


def classify_tweet(tweet: Dict[str, Any]) -> str:
    # priority: retweet > quote > reply > original
    if isinstance(tweet.get("retweeted_tweet"), dict):
        return "retweet"
    if isinstance(tweet.get("quoted_tweet"), dict):
        return "quote"
    if tweet.get("isReply") is True:
        return "reply"
    return "original"


def extract_media_from_one_tweet(tweet: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract media links from ONE tweet object:
    - photo -> media_url_https
    - video/animated_gif -> video_info.variants (mp4 + m3u8) + thumbnail
    """
    out: List[Dict[str, str]] = []
    media = (tweet.get("extendedEntities") or {}).get("media") or []

    for m in media:
        mtype = (m.get("type") or "").lower()

        if mtype == "photo":
            url = m.get("media_url_https") or m.get("media_url")
            if url:
                out.append({"kind": "image", "url": url})
            continue

        if mtype in ("video", "animated_gif"):
            vinfo = m.get("video_info") or {}
            variants = vinfo.get("variants") or []

            mp4s = [v.get("url") for v in variants if v.get("content_type") == "video/mp4" and v.get("url")]
            m3u8 = [v.get("url") for v in variants if "mpegURL" in (v.get("content_type") or "") and v.get("url")]

            for u in mp4s:
                out.append({"kind": "video" if mtype == "video" else "gif", "url": u})
            for u in m3u8:
                out.append({"kind": "hls", "url": u})

            thumb = m.get("media_url_https") or m.get("media_url")
            if thumb:
                out.append({"kind": "thumbnail", "url": thumb})

    return out


def extract_all_media(tweet: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract media from:
      - main tweet
      - quoted_tweet (if any)
      - retweeted_tweet (if any)
    """
    out: List[Dict[str, str]] = []
    out.extend(extract_media_from_one_tweet(tweet))

    qt = tweet.get("quoted_tweet")
    if isinstance(qt, dict):
        out.extend(extract_media_from_one_tweet(qt))

    rt = tweet.get("retweeted_tweet")
    if isinstance(rt, dict):
        out.extend(extract_media_from_one_tweet(rt))

    # dedup
    seen = set()
    deduped: List[Dict[str, str]] = []
    for item in out:
        key = (item.get("kind"), item.get("url"))
        if item.get("url") and key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped


# -----------------------------
# Timeline fetcher
# -----------------------------
def timeline_fetch_page(
    api_key: str,
    username: str,
    cursor: str,
    min_interval_s: float,
    timeout: int,
) -> Dict[str, Any]:
    # Use the param name that most commonly works for twitterapi.io
    params = {
        "userName": username,
    }
    if cursor:
        params["cursor"] = cursor

    return request_json(
        api_key=api_key,
        endpoint=TIMELINE_ENDPOINT,
        params=params,
        min_interval_s=min_interval_s,
        timeout=timeout,
    )


def extract_tweets_from_response(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Your actual response keys: status, code, msg, data, has_next_page, next_cursor
    Tweets are in resp["data"]["tweets"] (sometimes data could be list/items)
    """
    data_block = resp.get("data") or {}

    if isinstance(data_block, list):
        return data_block

    if isinstance(data_block, dict):
        for k in ("tweets", "items", "list"):
            v = data_block.get(k)
            if isinstance(v, list):
                return v

    return []


def fetch_user_timeline(
    api_key: str,
    username: str,
    min_interval_s: float,
    timeout: int,
    max_pages: int,
    max_tweets: int,
    since_utc: Optional[dt.datetime],
    exclude_replies: bool,
    retweets_only: bool,
    debug: bool,
) -> List[Dict[str, Any]]:
    cursor = ""
    all_tweets: List[Dict[str, Any]] = []
    seen_ids = set()

    for page_i in range(max_pages):
        resp = timeline_fetch_page(api_key, username, cursor, min_interval_s, timeout)

        if debug:
            print("[DEBUG raw keys]", resp.keys())

        tweets = extract_tweets_from_response(resp)

        if debug:
            print(f"[DEBUG] @{username} page {page_i+1}: got {len(tweets)} tweets (cursor='{cursor[:12]}...')")

        if not tweets:
            break

        for t in tweets:
            tid = str(t.get("id", ""))
            if not tid or tid in seen_ids:
                continue
            seen_ids.add(tid)

            created = parse_created_at(t)
            if since_utc and created and created < since_utc:
                # timeline is newest -> oldest, safe to stop
                return all_tweets

            ttype = classify_tweet(t)
            if exclude_replies and ttype == "reply":
                continue
            if retweets_only and ttype != "retweet":
                continue

            all_tweets.append(t)

            if len(all_tweets) >= max_tweets:
                return all_tweets

        # Pagination: your response provides these top-level keys
        has_next = bool(resp.get("has_next_page"))
        next_cursor = resp.get("next_cursor") or ""

        if not has_next or not next_cursor or next_cursor == cursor:
            break

        cursor = next_cursor

    return all_tweets


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--api_key", default=os.getenv("TWITTERAPI_IO_KEY"))
    ap.add_argument("--usernames", required=True, help="Path to username list file (one per line).")
    ap.add_argument("--out", default="timeline.jsonl", help="Output JSONL path")

    ap.add_argument("--days", type=int, default=0, help="If >0, keep tweets from last N days only (best-effort)")
    ap.add_argument("--max_pages", type=int, default=30, help="Max pages per user")
    ap.add_argument("--max_tweets_per_user", type=int, default=300, help="Max tweets kept per user (after filtering)")

    ap.add_argument("--exclude_replies", action="store_true", help="Exclude replies (keep original/quote/retweet)")
    ap.add_argument("--retweets_only", action="store_true", help="Only keep retweets/reposts")

    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout per request")
    ap.add_argument("--min_interval", type=float, default=DEFAULT_MIN_REQUEST_INTERVAL, help="Min seconds between requests")
    ap.add_argument("--debug", action="store_true", help="Print debug logs")

    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Provide --api_key or set TWITTERAPI_IO_KEY.")

    usernames = read_usernames(args.usernames)
    if not usernames:
        raise SystemExit("No usernames found in file.")

    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    since_utc: Optional[dt.datetime] = None
    if args.days and args.days > 0:
        since_utc = now - dt.timedelta(days=args.days)

    fetched_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    total_written = 0
    with open(args.out, "w", encoding="utf-8") as f_out:
        for i, u in enumerate(usernames, start=1):
            try:
                tweets = fetch_user_timeline(
                    api_key=args.api_key,
                    username=u,
                    min_interval_s=args.min_interval,
                    timeout=args.timeout,
                    max_pages=args.max_pages,
                    max_tweets=args.max_tweets_per_user,
                    since_utc=since_utc,
                    exclude_replies=args.exclude_replies,
                    retweets_only=args.retweets_only,
                    debug=args.debug,
                )

                for t in tweets:
                    line = {
                        "kol_username": u,
                        "fetched_at_utc": fetched_at,
                        "tweet_type": classify_tweet(t),
                        "created_at": t.get("createdAt"),
                        "media": extract_all_media(t),
                        "tweet": t,
                    }
                    f_out.write(json.dumps(line, ensure_ascii=False) + "\n")

                total_written += len(tweets)
                print(f"[{i}/{len(usernames)}] @{u}: wrote {len(tweets)} tweets")

            except Exception as e:
                print(f"[{i}/{len(usernames)}] @{u}: ERROR: {e}")

    print(f"Done. Wrote {total_written} tweets to {args.out}")


if __name__ == "__main__":
    main()
