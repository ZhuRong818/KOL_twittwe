

import os
import json
import time
import argparse
import datetime as dt
import random
import threading
import math
from typing import Any, Dict, List, Optional, Tuple

import requests
from dateutil import parser as dtparser
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor, as_completed


BASE_URL = "https://api.twitterapi.io"
SEARCH_ENDPOINT = "/twitter/tweet/advanced_search"
DEFAULT_TIMEOUT = 120


# -----------------------------
# Global QPS Limiter (token spacing)
# -----------------------------
class QpsLimiter:
    """Ensures total request rate across threads is <= qps (best-effort)."""
    def __init__(self, qps: float):
        if qps <= 0:
            raise ValueError("qps must be > 0")
        self.interval = 1.0 / qps
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self):
        with self.lock:
            now = time.perf_counter()
            if now < self.next_time:
                time.sleep(self.next_time - now)
            jitter = random.uniform(0.0, self.interval * 0.15)
            self.next_time = max(self.next_time, time.perf_counter()) + self.interval + jitter


class SkipUser401(Exception):
    """Skip a user after repeated 401 responses."""
    pass


class SkipUserLowEngagement(Exception):
    """Skip a user because sampled month engagement is below threshold."""
    pass


# -----------------------------
# Thread-local Session (connection reuse)
# -----------------------------
_thread_local = threading.local()

def get_session() -> requests.Session:
    """
    One Session per thread to reuse connections safely.
    (requests.Session is not guaranteed thread-safe if shared across threads)
    """
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        _thread_local.session = s
    return s


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
    if isinstance(tweet.get("retweeted_tweet"), dict):
        return "retweet"
    if isinstance(tweet.get("quoted_tweet"), dict):
        return "quote"
    if tweet.get("isReply") is True:
        return "reply"
    return "original"


def extract_media_from_one_tweet(tweet: Dict[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    media = (tweet.get("extendedEntities") or tweet.get("extended_entities") or {}).get("media") or []

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
    out: List[Dict[str, str]] = []
    out.extend(extract_media_from_one_tweet(tweet))

    qt = tweet.get("quoted_tweet")
    if isinstance(qt, dict):
        out.extend(extract_media_from_one_tweet(qt))

    rt = tweet.get("retweeted_tweet")
    if isinstance(rt, dict):
        out.extend(extract_media_from_one_tweet(rt))

    seen = set()
    deduped: List[Dict[str, str]] = []
    for item in out:
        key = (item.get("kind"), item.get("url"))
        if item.get("url") and key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def strip_extended_entities_fields(tweet: Dict[str, Any]) -> Dict[str, Any]:
    """Remove extendedEntities fields to reduce JSONL size; keep quoted/retweeted cleaned too."""
    if not isinstance(tweet, dict):
        return tweet

    cleaned = dict(tweet)
    cleaned.pop("extendedEntities", None)
    cleaned.pop("extended_entities", None)

    qt = cleaned.get("quoted_tweet")
    if isinstance(qt, dict):
        cleaned["quoted_tweet"] = strip_extended_entities_fields(qt)

    rt = cleaned.get("retweeted_tweet")
    if isinstance(rt, dict):
        cleaned["retweeted_tweet"] = strip_extended_entities_fields(rt)

    return cleaned

def fmt_utc_for_query(ts: dt.datetime) -> str:
    ts = ts.astimezone(dt.timezone.utc)
    return ts.strftime("%Y-%m-%d")


def month_slices(start_utc: dt.datetime, end_utc: dt.datetime) -> List[Tuple[dt.datetime, dt.datetime]]:
    slices: List[Tuple[dt.datetime, dt.datetime]] = []
    cur = start_utc.replace(microsecond=0)
    while cur < end_utc:
        nxt = (cur + relativedelta(months=1)).replace(microsecond=0)
        if nxt > end_utc:
            nxt = end_utc
        slices.append((cur, nxt))
        cur = nxt
    return slices


# -----------------------------
# Engagement (your formula)
# eng = likes + 2*retweets + 2*quotes + 3*replies + 1*log10(views+1)
# -----------------------------
def _to_int(v: Any) -> int:
    if isinstance(v, bool):
        return 0
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            return int(s)
        # try parse float-like
        try:
            return int(float(s))
        except Exception:
            return 0
    return 0


def get_count(tweet: Dict[str, Any], keys: List[str]) -> int:
    for k in keys:
        if k in tweet:
            return max(0, _to_int(tweet.get(k)))
    return 0


def estimate_views(tweet: Dict[str, Any]) -> int:
    """
    Provider-dependent. We try multiple common keys; if none exist, views=0.
    """
    keys = [
        "viewCount", "views", "viewsCount",
        "impressionCount", "impressions", "impressionsCount",
        "playCount",  # sometimes used for video views
    ]
    return get_count(tweet, keys)


def tweet_engagement(tweet: Dict[str, Any]) -> float:
    likes = get_count(tweet, ["likeCount", "favorite_count", "favouritesCount", "favorites", "likes"])
    rts = get_count(tweet, ["retweetCount", "retweet_count", "retweets", "reposts"])
    quotes = get_count(tweet, ["quoteCount", "quote_count", "quotes"])
    replies = get_count(tweet, ["replyCount", "reply_count", "replies"])
    views = estimate_views(tweet)

    eng = (
        likes
        + 2 * rts
        + 2 * quotes
        + 3 * replies
        + 2.0 * math.log10(views + 1.0)
    )
    return eng


def avg_engagement(total: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return total / float(count)


# -----------------------------
# Response parsing (tolerant)
# -----------------------------
def safe_get_tweets(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    v = resp.get("tweets")
    if isinstance(v, list):
        return v

    data = resp.get("data")
    if isinstance(data, dict):
        for k in ("tweets", "items", "list"):
            vv = data.get(k)
            if isinstance(vv, list):
                return vv

    print("[WARN] cannot find tweets in response. top_keys=", list(resp.keys())[:20])
    return []


def safe_get_pageinfo(resp: Dict[str, Any]) -> Tuple[bool, str]:
    if "has_next_page" in resp or "next_cursor" in resp:
        return bool(resp.get("has_next_page")), (resp.get("next_cursor") or "")

    data = resp.get("data")
    if isinstance(data, dict) and ("has_next_page" in data or "next_cursor" in data):
        return bool(data.get("has_next_page")), (data.get("next_cursor") or "")

    print("[WARN] cannot find pagination keys in response. top_keys=", list(resp.keys())[:20])
    return False, ""


# -----------------------------
# Request helper (with limiter)
# -----------------------------
def request_json(
    api_key: str,
    endpoint: str,
    params: Dict[str, Any],
    limiter: QpsLimiter,
    timeout: int,
    max_retries: int = 5,
) -> Dict[str, Any]:
    url = f"{BASE_URL}{endpoint}"
    headers = {"X-API-Key": api_key}

    for attempt in range(1, max_retries + 1):
        limiter.wait()
        try:
            session = get_session()
            r = session.get(url, headers=headers, params=params, timeout=timeout)

            if r.status_code == 401:
                body = (r.text or "")[:400]
                if attempt < 2:
                    time.sleep(1.0 + random.random())
                    continue
                raise SkipUser401(f"HTTP 401: {body}")

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                backoff = float(retry_after) if retry_after else (2 ** min(attempt, 6)) + random.random()
                print(f"[WARN] 429 rate limited. Sleep {backoff:.1f}s then retry ({attempt}/{max_retries})")
                time.sleep(backoff)
                continue

            if 500 <= r.status_code < 600:
                if attempt < max_retries:
                    backoff = (2 ** min(attempt, 6)) + random.random()
                    print(f"[WARN] HTTP {r.status_code}. Sleep {backoff:.1f}s then retry ({attempt}/{max_retries})")
                    time.sleep(backoff)
                    continue

            if r.status_code != 200:
                body = (r.text or "")[:400]
                raise RuntimeError(f"HTTP {r.status_code}: {body}")

            return r.json()

        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            if attempt >= max_retries:
                raise RuntimeError(f"Network error after retries: {e}") from e
            backoff = (2 ** min(attempt, 6)) + random.random()
            print(f"[WARN] {type(e).__name__}. Sleep {backoff:.1f}s then retry ({attempt}/{max_retries})")
            time.sleep(backoff)

    raise RuntimeError("request_json failed unexpectedly")


# -----------------------------
# Advanced Search fetcher
# -----------------------------
def search_fetch_page(
    api_key: str,
    query: str,
    cursor: str,
    limiter: QpsLimiter,
    timeout: int,
) -> Dict[str, Any]:
    params = {
        "query": query,
        "queryType": "Latest",
        "cursor": cursor or "",
    }
    return request_json(api_key, SEARCH_ENDPOINT, params, limiter, timeout)


def fetch_user_5y(
    api_key: str,
    username: str,
    limiter: QpsLimiter,
    timeout: int,
    exclude_replies: bool,
    retweets_only: bool,
    exclude_retweets: bool,
    debug: bool,
    max_pages_per_slice: int,
    deep_pages_per_slice: int,
    ultra_deep_pages_per_slice: int,
    engagement_threshold_deep: float,
    ultra_engagement_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Engagement gate behavior:
    - We sample each month up to max_pages_per_slice (typically 40).
    - If a month requires >max_pages_per_slice pages (has_next_page still true after sampling),
      compute month_engagement from sampled tweets (pages 1..max_pages_per_slice).
    - For content-heavy months (> max_pages_per_slice):
        * check this month + the most recent 3 months; if any month avg_eng >= threshold -> unlock deep fetch
        * else skip ENTIRE user
    - For normal months (<= max_pages_per_slice), no engagement gate.
    """
    now_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    start_utc = (now_utc - relativedelta(years=5)).replace(microsecond=0)

    # Process newest month first so engagement gating evaluates recent activity first
    slices = list(reversed(month_slices(start_utc, now_utc)))

    all_tweets: List[Dict[str, Any]] = []
    seen_ids = set()

    user_deep_unlocked = False  # once unlocked, we fetch deeper for rest of user
    user_deep_cap = deep_pages_per_slice
    recent_month_avgs: List[float] = []  # newest -> older; store last 3 completed months
    for (since_dt, until_dt) in slices:
        since_q = fmt_utc_for_query(since_dt)
        until_q = fmt_utc_for_query(until_dt)
        query = f"from:{username} since:{since_q} until:{until_q}"

        cursor = ""
        page = 0
        seen_cursors = set()
        month_eng = 0.0
        month_eng_count = 0
        last_has_next = False

        while True:
            page += 1

            cap = user_deep_cap if user_deep_unlocked else max_pages_per_slice
            if page > cap:
                # Stop this slice (controlled omission)
                if not user_deep_unlocked and cap == max_pages_per_slice:
                    print(f"[WARN] slice too deep, stop early (sampling cap): @{username} {since_q}~{until_q} pages={page-1} eng={month_eng:.2f}")
                else:
                    print(f"[WARN] slice too deep, stop early (deep cap): @{username} {since_q}~{until_q} pages={page-1} eng={month_eng:.2f}")
                break

            resp = search_fetch_page(api_key, query, cursor, limiter, timeout)
            tweets = safe_get_tweets(resp)

            if debug:
                print(f"[DEBUG] @{username} slice {since_q}~{until_q} page {page}: {len(tweets)} cursor='{(cursor or '')[:10]}...'")

            # Accumulate month engagement only for sampling phase (before unlock)
            for t in tweets:
                tid = str(t.get("id", ""))
                if not tid or tid in seen_ids:
                    continue
                seen_ids.add(tid)

                ttype = classify_tweet(t)
                if exclude_replies and ttype == "reply":
                    continue
                if exclude_retweets and ttype == "retweet":
                    continue
                if retweets_only and ttype != "retweet":
                    continue

                # Engagement is computed on the filtered tweets we actually keep
                if not user_deep_unlocked and page <= max_pages_per_slice:
                    month_eng += tweet_engagement(t)
                    month_eng_count += 1

                all_tweets.append(t)

            has_next, next_cursor = safe_get_pageinfo(resp)
            last_has_next = has_next

            if not has_next or not next_cursor or next_cursor == cursor:
                break

            if next_cursor in seen_cursors:
                print(f"[WARN] cursor loop detected, stop slice early: @{username} {since_q}~{until_q}")
                break
            seen_cursors.add(next_cursor)

            # ---- Engagement Gate Trigger ----
            # We only gate when:
            # 1) user not unlocked yet
            # 2) we just finished sampling page = max_pages_per_slice
            # 3) there ARE more pages (has_next True)
            if (not user_deep_unlocked) and (page == max_pages_per_slice) and has_next:
                month_avg = avg_engagement(month_eng, month_eng_count)
                window_avgs = recent_month_avgs[-3:] + [month_avg]
                if any(a >= engagement_threshold_deep for a in window_avgs):
                    max_avg = max(window_avgs)
                    if max_avg >= ultra_engagement_threshold:
                        user_deep_cap = ultra_deep_pages_per_slice
                    user_deep_unlocked = True
                    print(
                        f"[INFO] deep unlocked: @{username} month {since_q}~{until_q} "
                        f"window_avg_max={max_avg:.2f} >= {engagement_threshold_deep} (cap -> {user_deep_cap})"
                    )
                else:
                    raise SkipUserLowEngagement(
                        f"@{username} month {since_q}~{until_q} needs >{max_pages_per_slice} pages "
                        f"and recent window avg max={max(window_avgs):.2f} < threshold={engagement_threshold_deep}"
                    )

            cursor = next_cursor

        # record avg engagement for this completed month (sampling only)
        if not user_deep_unlocked:
            recent_month_avgs.append(avg_engagement(month_eng, month_eng_count))

    return all_tweets


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", default=os.getenv("TWITTERAPI_IO_KEY"))
    ap.add_argument("--usernames", required=True, help="Path to username list file (one per line).")
    ap.add_argument("--out", default="final_data/search_5y.jsonl", help="Output JSONL path")
    ap.add_argument("--failed_out", default="final_data/failed_users.txt", help="Failed users output (one per line)")

    ap.add_argument("--workers", type=int, default=12, help="Threads for user-level parallelism")
    ap.add_argument("--qps", type=float, default=18.0, help="Total QPS across all workers (use < 20 for stability)")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)

    ap.add_argument("--exclude_replies", action="store_true")
    ap.add_argument("--retweets_only", action="store_true")
    ap.add_argument("--exclude_retweets", action="store_true")

    ap.add_argument(
        "--strip_extended_entities",
        action="store_true",
        help="Remove tweet.extendedEntities/extended_entities from output JSONL (media links still extracted).",
    )

    # Sampling  per month
    ap.add_argument("--max_pages_per_slice", type=int, default=30, help="Sampling cap pages per month slice (~20 tweets/page)")

    # If unlocked, allow deeper per month
    ap.add_argument("--deep_pages_per_slice", type=int, default=300, help="Deep cap pages per month slice after unlock")
    ap.add_argument("--ultra_deep_pages_per_slice", type=int, default=450, help="Ultra deep cap pages per month slice after unlock")

    # Engagement threshold gate (per-tweet average) for content-heavy months only
    ap.add_argument(
        "--engagement_threshold_deep",
        type=float,
        default=140.0,
        help="If a month exceeds max_pages_per_slice, require avg engagement >= this to unlock deep fetch.",
    )
    ap.add_argument(
        "--ultra_engagement_threshold",
        type=float,
        default=500.0,
        help="If recent window avg engagement >= this, unlock ultra deep cap.",
    )

    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    args.api_key = (args.api_key or "").strip()
    if not args.api_key:
        raise SystemExit("Missing API key. Provide --api_key or set TWITTERAPI_IO_KEY.")

    # Ensure outputs go to final_data/ when a bare filename is provided.
    if os.path.dirname(args.out) == "":
        args.out = os.path.join("final_data", args.out)

    # If failed_out not customized, derive from usernames file prefix (before first "_")
    if args.failed_out == "final_data/failed_users.txt":
        base = os.path.basename(args.usernames)
        prefix = base.split("_", 1)[0] if "_" in base else os.path.splitext(base)[0]
        args.failed_out = os.path.join("final_data", f"{prefix}_failed_users.txt")
    elif os.path.dirname(args.failed_out) == "":
        args.failed_out = os.path.join("final_data", args.failed_out)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.failed_out), exist_ok=True)

    usernames = read_usernames(args.usernames)
    if not usernames:
        raise SystemExit("No usernames found in file.")

    limiter = QpsLimiter(qps=args.qps)
    fetched_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")

    total_written = 0
    failed_users: List[str] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        future_to_user = {
            ex.submit(
                fetch_user_5y,
                args.api_key,
                u,
                limiter,
                args.timeout,
                args.exclude_replies,
                args.retweets_only,
                args.exclude_retweets,
                args.debug,
                args.max_pages_per_slice,
                args.deep_pages_per_slice,
                args.ultra_deep_pages_per_slice,
                args.engagement_threshold_deep,
                args.ultra_engagement_threshold,
            ): u
            for u in usernames
        }

        with open(args.out, "w", encoding="utf-8") as f_out:
            done_count = 0
            n = len(usernames)

            for fut in as_completed(future_to_user):
                u = future_to_user[fut]
                done_count += 1
                try:
                    tweets = fut.result()

                    for t in tweets:
                        media = extract_all_media(t)
                        cleaned_tweet = t
                        if args.strip_extended_entities and isinstance(t, dict):
                            cleaned_tweet = strip_extended_entities_fields(t)

                        line = {
                            "kol_username": u,
                            "fetched_at_utc": fetched_at,
                            "tweet_type": classify_tweet(t),
                            "created_at": t.get("createdAt") or t.get("created_at"),
                            "media": media,
                            "tweet": cleaned_tweet,
                        }
                        f_out.write(json.dumps(line, ensure_ascii=False) + "\n")

                    total_written += len(tweets)
                    print(f"[{done_count}/{n}] @{u}: wrote {len(tweets)} tweets")

                except SkipUser401 as e:
                    failed_users.append(u)
                    print(f"[{done_count}/{n}] @{u}: SKIP (401) {e}")

                except SkipUserLowEngagement as e:
                    failed_users.append(u)
                    print(f"[{done_count}/{n}] @{u}: SKIP (low engagement) {e}")

                except Exception as e:
                    failed_users.append(u)
                    print(f"[{done_count}/{n}] @{u}: ERROR {e}")

    if failed_users:
        seen = set()
        dedup = []
        for u in failed_users:
            if u.lower() not in seen:
                seen.add(u.lower())
                dedup.append(u)

        with open(args.failed_out, "w", encoding="utf-8") as f:
            for u in dedup:
                f.write(u + "\n")

        print(f"[WARN] Failed users: {len(dedup)} written to {args.failed_out}")

    print(f"Done. Wrote {total_written} tweets to {args.out}")


if __name__ == "__main__":
    main()
