import csv
import math
import os
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 配置区
# =========================
# ✅ 建议用环境变量：export TWITTERAPI_IO_KEY="xxx"
API_KEY = os.getenv("TWITTERAPI_IO_KEY", "").strip() or "PUT_YOUR_KEY_HERE"
BASE = "https://api.twitterapi.io"

INPUT_HANDLES_FILE = "handles.txt"

TWEETS_PER_USER = 20
MIN_FOLLOWERS = 200_000

W_SIM = 0.5
W_ENG = 0.5

TOP_N_OUTPUT = 300
OUT_RANKED_CSV = "sports_full.csv"
OUT_TOP_TXT = "sports_top300.txt"

SLEEP_SEC = 0.25
RETRY_429_SLEEP_SEC = 6.0
TIMEOUT_SEC = 30


# =========================
# Sports 关键词（完整版）
# =========================
CORE_SPORTS = [
    "football", "basketball", "baseball",
    "soccer", "tennis", "golf",
    "hockey", "cricket", "rugby",
    "mma", "boxing", "esports",
    "athletics", "swimming", "cycling",
    "gymnastics", "volleyball", "badminton",
    "table tennis", "wrestling", "motorsports"
]

DEV_ECOSYSTEM = [
    "developer tools", "developer experience", "dx",
    "open source", "maintainer", "contributor",
    "sdk", "api", "framework", "library",
    "tooling", "cli", "build tools",
    "ci cd", "deployment", "release pipeline",
]

LANG_RUNTIME = [
    "programming language", "runtime", "compiler", "interpreter",
    "python", "javascript", "typescript", "java", "go", "golang",
    "rust", "c++", "c#", "swift", "kotlin", "php",
    "node.js", "deno", "bun", "jvm",
]

AI_DATA = [
    "artificial intelligence", "machine learning", "deep learning",
    "foundation model", "large language model", "llm",
    "model training", "inference", "fine tuning",
    "data science", "data engineering", "data pipeline",
    "vector database", "embedding", "rag",
    "analytics", "experimentation", "ab testing",
]

TECH_PRODUCT = [
    "product engineering", "technical roadmap",
    "platform strategy", "architecture decision",
    "build vs buy", "technical debt",
    "scaling a product", "engineering productivity",
    "developer platform", "internal tooling",
    "saas platform", "b2b software",
]

STARTUP_TECH = [
    "tech startup", "founder engineering",
    "early stage startup", "y combinator",
    "product market fit", "pmf",
    "tech stack", "scaling team",
    "engineering culture", "hiring engineers",
]

SECURITY_INFRA = [
    "cybersecurity", "security engineering",
    "privacy", "data protection",
    "authentication", "authorization",
    "infrastructure security",
    "cloud security", "zero trust",
]

TECH_TRENDS = [
    "technology trends", "tech industry", "tech news",
    "innovation", "digital transformation",
    "emerging technology", "future of technology",
    "disruptive technology",
    "big tech", "tech giants",
    "silicon valley", "startup ecosystem",
]

CONSUMER_TECH = [
    "consumer technology", "tech products",
    "smartphone", "mobile app", "app launch",
    "hardware", "wearables", "smart devices",
    "user experience", "ux", "ui",
    "product design", "product launch",
]



SPORTS_KEYPHRASES = (
    CORE_SPORTS + DEV_ECOSYSTEM + LANG_RUNTIME + AI_DATA
    + TECH_PRODUCT + STARTUP_TECH + SECURITY_INFRA
    + TECH_TRENDS + CONSUMER_TECH
)


# 可选：垃圾内容惩罚（和你之前版本一致）
SPAM_PATTERNS = [
    r"\b(sign up|join|register)\b",
    r"\blink in bio\b",
    r"\bsubscribe\b",
    r"\bgiveaway\b",
    r"\bpromo\b|\bdiscount\b",
    r"\bwe'?re hiring\b|\bhiring\b",
]


# =========================
# 数据结构
# =========================
@dataclass
class UserProfile:
    username: str
    followers: int


@dataclass
class Tweet:
    text: str
    likeCount: int
    replyCount: int
    retweetCount: int
    quoteCount: int


@dataclass
class UserScore:
    username: str
    followers: int
    eligible: bool

    sim_raw: float
    eng_raw: float
    spam_ratio: float
    rt_ratio: float

    sim_norm: float
    eng_norm: float
    score: float

    error: str = ""


# =========================
# 工具函数
# =========================
def load_handles(path: str) -> List[str]:
    handles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            h = line.strip()
            if not h:
                continue
            if h.startswith("@"):
                h = h[1:]
            handles.append(h)

    # 去重但保持顺序
    seen = set()
    uniq = []
    for h in handles:
        if h not in seen:
            seen.add(h)
            uniq.append(h)
    return uniq


def safe_get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not API_KEY or API_KEY == "PUT_YOUR_KEY_HERE":
        raise RuntimeError("API_KEY not set. Set TWITTERAPI_IO_KEY env var or edit API_KEY.")

    headers = {"X-API-Key": API_KEY}
    last_err = None

    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT_SEC)
            if r.status_code == 429:
                time.sleep(RETRY_429_SLEEP_SEC * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(1)

    raise RuntimeError(f"API request failed: {last_err}")


def _find_first_int(obj: Any, keys_lower: set) -> int:
    """递归查找 followers 字段，避免结构变化导致 followers=0"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if lk in keys_lower:
                try:
                    return int(v)
                except Exception:
                    pass
            got = _find_first_int(v, keys_lower)
            if got is not None and got != -1:
                return got
    elif isinstance(obj, list):
        for it in obj:
            got = _find_first_int(it, keys_lower)
            if got is not None and got != -1:
                return got
    return -1


def fetch_user_info(username: str) -> UserProfile:
    data = safe_get(f"{BASE}/twitter/user/info", params={"userName": username})
    keys_lower = {
        "followers", "followerscount", "followers_count",
        "followercount", "follower_count"
    }
    followers = _find_first_int(data, keys_lower)
    if followers == -1:
        followers = 0
    return UserProfile(username=username, followers=int(followers))


def fetch_last_tweets(username: str, limit: int, include_replies: bool = False) -> List[Tweet]:
    data = safe_get(
        f"{BASE}/twitter/user/last_tweets",
        params={"userName": username, "includeReplies": str(include_replies).lower(), "cursor": ""}
    )
    tweets_raw = (data.get("tweets") or [])[:limit]

    out: List[Tweet] = []
    for t in tweets_raw:
        out.append(Tweet(
            text=(t.get("text") or "").strip(),
            likeCount=int(t.get("likeCount") or 0),
            replyCount=int(t.get("replyCount") or 0),
            retweetCount=int(t.get("retweetCount") or 0),
            quoteCount=int(t.get("quoteCount") or 0),
        ))
    return out


def compute_similarity_avg(texts: List[str], keyphrases: List[str]) -> float:
    texts = [t for t in texts if t]
    if not texts:
        return 0.0

    docs = texts + keyphrases
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1
    )
    X = vectorizer.fit_transform(docs)
    X_text = X[:len(texts)]
    X_key = X[len(texts):]

    sims = cosine_similarity(X_text, X_key)
    max_per_tweet = sims.max(axis=1)
    return float(max_per_tweet.mean())


def compute_engagement_raw(tweets: List[Tweet], followers: int) -> float:
    if not tweets:
        return 0.0
    f = max(followers, 1)
    ers = []
    for t in tweets:
        inter = t.likeCount + t.replyCount + t.retweetCount + t.quoteCount
        ers.append(inter / f)
    er_avg = sum(ers) / len(ers)
    return float(math.log1p(er_avg * 1000.0))


def compute_spam_rt(texts: List[str]) -> Tuple[float, float]:
    if not texts:
        return 0.0, 0.0

    spam = 0
    rt = 0
    for tx in texts:
        low = tx.lower()
        if tx.startswith("RT @"):
            rt += 1
        if any(re.search(p, low) for p in SPAM_PATTERNS):
            spam += 1

    n = max(len(texts), 1)
    return spam / n, rt / n


def minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


# =========================
# 主流程
# =========================
def main():
    handles = load_handles(INPUT_HANDLES_FILE)
    print(f"Loaded {len(handles)} handles from {INPUT_HANDLES_FILE}")

    # 1) 拉 profile（followers）
    profiles: Dict[str, UserProfile] = {}
    for i, u in enumerate(handles, 1):
        try:
            profiles[u] = fetch_user_info(u)
        except Exception as e:
            profiles[u] = UserProfile(username=u, followers=0)
            print(f"[WARN] user/info failed for {u}: {e}")

        time.sleep(SLEEP_SEC)
        if i % 100 == 0:
            print(f"Fetched profiles: {i}/{len(handles)}")

    # 2) 对 eligible 拉 tweets 并计算 raw 分
    temp: List[UserScore] = []
    for i, u in enumerate(handles, 1):
        p = profiles[u]
        eligible = p.followers >= MIN_FOLLOWERS

        if not eligible:
            temp.append(UserScore(
                username=u, followers=p.followers, eligible=False,
                sim_raw=0.0, eng_raw=0.0, spam_ratio=0.0, rt_ratio=0.0,
                sim_norm=0.0, eng_norm=0.0, score=-1e9,
                error="followers_below_threshold"
            ))
            continue

        try:
            tweets = fetch_last_tweets(u, limit=TWEETS_PER_USER, include_replies=False)
            texts = [t.text for t in tweets]

            sim_raw = compute_similarity_avg(texts, TECH_KEYPHRASES)
            eng_raw = compute_engagement_raw(tweets, p.followers)
            spam_r, rt_r = compute_spam_rt(texts)

            temp.append(UserScore(
                username=u, followers=p.followers, eligible=True,
                sim_raw=sim_raw, eng_raw=eng_raw,
                spam_ratio=spam_r, rt_ratio=rt_r,
                sim_norm=0.0, eng_norm=0.0, score=0.0,
                error=""
            ))
        except Exception as e:
            temp.append(UserScore(
                username=u, followers=p.followers, eligible=True,
                sim_raw=0.0, eng_raw=0.0, spam_ratio=0.0, rt_ratio=0.0,
                sim_norm=0.0, eng_norm=0.0, score=-1e9,
                error=str(e)
            ))

        time.sleep(SLEEP_SEC)
        if i % 100 == 0:
            print(f"Scored users: {i}/{len(handles)}")

    # 3) eligible 且无错误的做归一化
    eligible_idx = [idx for idx, s in enumerate(temp) if s.eligible and not s.error and s.score > -1e8]
    sim_list = [temp[idx].sim_raw for idx in eligible_idx]
    eng_list = [temp[idx].eng_raw for idx in eligible_idx]

    sim_norm = minmax_norm(sim_list)
    eng_norm = minmax_norm(eng_list)

    for k, idx in enumerate(eligible_idx):
        s = temp[idx]
        s.sim_norm = float(sim_norm[k])
        s.eng_norm = float(eng_norm[k])

        # 可选惩罚（你要严格按 0.4/0.6，也可以把惩罚去掉）
        penalty = 0.2 * s.spam_ratio + 0.15 * s.rt_ratio
        s.score = float(W_SIM * s.sim_norm + W_ENG * s.eng_norm - penalty)

    # 4) 排序：good 在前
    def sort_key(s: UserScore) -> Tuple[int, float]:
        good = (s.eligible and not s.error and s.score > -1e8)
        group = 0 if good else 1
        return (group, -s.score)

    ranked = sorted(temp, key=sort_key)

    # 5) 输出全量 CSV
    with open(OUT_RANKED_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "username", "followers", "eligible",
            "sim_raw", "eng_raw", "sim_norm", "eng_norm",
            "spam_ratio", "rt_ratio",
            "score", "error"
        ])
        for r, s in enumerate(ranked, 1):
            w.writerow([
                r, s.username, s.followers, s.eligible,
                f"{s.sim_raw:.6f}", f"{s.eng_raw:.6f}",
                f"{s.sim_norm:.6f}", f"{s.eng_norm:.6f}",
                f"{s.spam_ratio:.3f}", f"{s.rt_ratio:.3f}",
                f"{s.score:.6f}", s.error
            ])

    # 6) 输出 topN handles
    top_good = [s for s in ranked if s.eligible and not s.error and s.score > -1e8]
    topN = top_good[:TOP_N_OUTPUT]

    with open(OUT_TOP_TXT, "w", encoding="utf-8") as f:
        for s in topN:
            f.write(s.username + "\n")

    print(f"Saved full ranking: {OUT_RANKED_CSV}")
    print(f"Saved top handles: {OUT_TOP_TXT}")
    print(f"Eligible good users: {len(top_good)} / {len(handles)}")


if __name__ == "__main__":
    main()
