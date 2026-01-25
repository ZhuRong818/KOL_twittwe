import csv
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple

import requests
import numpy as np
from dateutil import parser as dtparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 配置区（你只需要改这里）
# =========================

API_KEY = os.getenv("TWITTERAPI_IO_KEY")
if not API_KEY:
    raise SystemExit("Missing TWITTERAPI_IO_KEY")

BASE = "https://api.twitterapi.io"

INPUT_HANDLES_FILE = "health_seeds.txt"

# 省钱：每个账号只拉 N 条（注意：twitterapi.io 的 last_tweets 默认可能返回固定数量，这里只截断）
TWEETS_PER_USER = 20

# 你的旧逻辑：只看最近 N 天
LOOKBACK_DAYS = 60

# 你的旧逻辑：最终只取 TOP_K
TOP_K = 250

# 输出
OUT_RANKED_CSV = "health_ranked_full.csv"
OUT_TOP_TXT = "health_top250.txt"

# 请求节流：避免 429
SLEEP_SEC = 0.30
RETRY_429_SLEEP_SEC = 5.0
TIMEOUT_SEC = 60


# ========= 健康领域关键词（原样保留） =========
HEALTH_KEYWORDS = [
    "health", "public health", "medicine", "medical", "doctor", "physician",
    "epidemiology", "epidemic", "pandemic", "covid",
    "vaccine", "vaccination", "virus", "disease",
    "clinical", "trial", "nutrition", "diet",
    "mental health", "healthcare", "hospital", "drug",
    "wellness", "fitness", "exercise", "obesity",
    "health policy", "health equity", "global health",
    "health education", "health promotion", "health communication",
    "chronic disease", "infectious disease",
    "immunology", "health disparities",
    "health system", "health services", "health research", "illness",
    "pharmaceutical", "therapy", "treatment", "cancer",
    "diabetes", "cardiovascular", "nutrition", "allergy",
    "mental illness", "depression", "anxiety",
    "addiction", "substance abuse",
    "disease prevention"
]

# ========= 硬过滤（原样保留） =========
MIN_FOLLOWERS = 100000
MIN_TWEETS = 8
MIN_ORIGINAL_RATIO = 0.4
MIN_MEDIAN_VIEWS = 15000
MIN_RELEVANCE = 0.08


# =========================
# 数据结构
# =========================
@dataclass
class UserProfile:
    username: str
    description: str
    followers: int


@dataclass
class Tweet:
    text: str
    createdAt: str
    viewCount: int
    likeCount: int
    replyCount: int
    retweetCount: int
    quoteCount: int


@dataclass
class UserScore:
    username: str
    eligible: bool

    # 过滤/特征
    median_views: float
    median_engagement: float
    active_days: int
    originality: float

    # 主题相关
    rel_lex: float         # 你的旧 relevance_score（词典命中）
    sim_tfidf: float       # 新模板风格：TFIDF 相似度（可选但很有用）

    # 归一化 + 最终分
    views_norm: float
    eng_norm: float
    rel_norm: float
    orig_norm: float
    act_norm: float
    sim_norm: float
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
    headers = {"X-API-Key": API_KEY}
    for i in range(6):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT_SEC)
            if r.status_code == 429:
                time.sleep(RETRY_429_SLEEP_SEC + i * 2.0)
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(min(2 ** i, 30))
    raise RuntimeError(f"Failed request: {url}")


def fetch_user_info(username: str) -> UserProfile:
    data = safe_get(f"{BASE}/twitter/user/info", params={"userName": username})
    user = data.get("user") or data.get("data") or data
    if isinstance(data.get("data"), dict) and isinstance(data["data"].get("data"), dict):
        user = data["data"]["data"]
    desc = (user.get("description") or "").strip()
    followers = user.get("followersCount") or user.get("followers_count") or user.get("followers") or 0
    return UserProfile(username=username, description=desc, followers=int(followers or 0))


def fetch_last_tweets(username: str, limit: int, include_replies: bool = False) -> List[Tweet]:
    data = safe_get(
        f"{BASE}/twitter/user/last_tweets",
        params={"userName": username, "includeReplies": str(include_replies).lower(), "cursor": ""}
    )
    if isinstance(data.get("data"), dict) and isinstance(data["data"].get("tweets"), list):
        tweets_raw = data["data"]["tweets"]
    else:
        tweets_raw = data.get("tweets", []) or []
    tweets_raw = tweets_raw[:limit]

    out: List[Tweet] = []
    for t in tweets_raw:
        out.append(Tweet(
            text=(t.get("text") or "").strip(),
            createdAt=(t.get("createdAt") or ""),
            viewCount=int(t.get("viewCount") or 0),
            likeCount=int(t.get("likeCount") or 0),
            replyCount=int(t.get("replyCount") or 0),
            retweetCount=int(t.get("retweetCount") or 0),
            quoteCount=int(t.get("quoteCount") or 0),
        ))
    return out


def is_retweet_text(text: str) -> bool:
    text = (text or "").lower()
    return text.startswith("rt @")


def relevance_score_lexicon(bio: str, tweets: List[Tweet]) -> float:
    """
    复刻你旧代码的 relevance_score：bio 命中更重，tweet 命中较轻。
    最后压到 [0,1]
    """
    bio_l = (bio or "").lower()
    score = sum(1 for k in HEALTH_KEYWORDS if k in bio_l) * 1.2
    for t in tweets:
        txt = (t.text or "").lower()
        score += sum(1 for k in HEALTH_KEYWORDS if k in txt) * 0.25
    return float(min(score / 12.0, 1.0))


def compute_similarity_avg_tfidf(texts: List[str], keyphrases: List[str]) -> float:
    """
    模板里的 TF-IDF + cosine：
    每条推文对 keyphrases 求 cosine，相似度取 max；再对推文取平均。
    """
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


def extract_engagement(t: Tweet) -> int:
    return int(t.likeCount + t.replyCount + t.retweetCount + t.quoteCount)


def parse_created_at(ts: str) -> datetime:
    dt = dtparser.parse(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


# =========================
# 单个 KOL 打分（对齐你旧逻辑，但结构像模板）
# =========================
def score_health_kol(username: str) -> UserScore:
    try:
        prof = fetch_user_info(username)
        if prof.followers < MIN_FOLLOWERS:
            return UserScore(
                username=username, eligible=False,
                median_views=0.0, median_engagement=0.0, active_days=0, originality=0.0,
                rel_lex=0.0, sim_tfidf=0.0,
                views_norm=0.0, eng_norm=0.0, rel_norm=0.0, orig_norm=0.0, act_norm=0.0, sim_norm=0.0,
                score=-1e9, error="min_followers"
            )
        tweets = fetch_last_tweets(username, limit=TWEETS_PER_USER, include_replies=False)

        cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)

        views: List[int] = []
        engs: List[int] = []
        active_days = set()
        originals = 0

        for t in tweets:
            if not t.createdAt:
                continue
            dt = parse_created_at(t.createdAt)
            if dt < cutoff:
                continue

            views.append(int(t.viewCount or 0))
            engs.append(extract_engagement(t))
            active_days.add(dt.date())

            if not is_retweet_text(t.text):
                originals += 1

        # ========= 过滤（原逻辑） =========
        if len(views) < MIN_TWEETS:
            return UserScore(
                username=username, eligible=False,
                median_views=0.0, median_engagement=0.0, active_days=0, originality=0.0,
                rel_lex=0.0, sim_tfidf=0.0,
                views_norm=0.0, eng_norm=0.0, rel_norm=0.0, orig_norm=0.0, act_norm=0.0, sim_norm=0.0,
                score=-1e9, error="min_tweets"
            )

        originality = originals / max(len(views), 1)
        if originality < MIN_ORIGINAL_RATIO:
            return UserScore(
                username=username, eligible=False,
                median_views=0.0, median_engagement=0.0, active_days=len(active_days), originality=originality,
                rel_lex=0.0, sim_tfidf=0.0,
                views_norm=0.0, eng_norm=0.0, rel_norm=0.0, orig_norm=0.0, act_norm=0.0, sim_norm=0.0,
                score=-1e9, error="min_original_ratio"
            )

        median_views = float(np.median(views))
        if median_views < MIN_MEDIAN_VIEWS:
            return UserScore(
                username=username, eligible=False,
                median_views=median_views, median_engagement=0.0, active_days=len(active_days), originality=originality,
                rel_lex=0.0, sim_tfidf=0.0,
                views_norm=0.0, eng_norm=0.0, rel_norm=0.0, orig_norm=0.0, act_norm=0.0, sim_norm=0.0,
                score=-1e9, error="min_median_views"
            )

        median_eng = float(np.median(engs))
        rel_lex = relevance_score_lexicon(prof.description, tweets)
        if rel_lex < MIN_RELEVANCE:
            return UserScore(
                username=username, eligible=False,
                median_views=median_views, median_engagement=median_eng, active_days=len(active_days), originality=originality,
                rel_lex=rel_lex, sim_tfidf=0.0,
                views_norm=0.0, eng_norm=0.0, rel_norm=0.0, orig_norm=0.0, act_norm=0.0, sim_norm=0.0,
                score=-1e9, error="min_relevance"
            )

        # ========= 新增：TF-IDF 相似度（模板风格，增强 topic 判别） =========
        texts = [t.text for t in tweets if t.text]
        sim_tfidf = compute_similarity_avg_tfidf(texts, HEALTH_KEYWORDS)

        # 先把 eligible 的特征填上，norm/score 之后主流程再统一算
        return UserScore(
            username=username, eligible=True,
            median_views=median_views,
            median_engagement=median_eng,
            active_days=len(active_days),
            originality=float(originality),
            rel_lex=float(rel_lex),
            sim_tfidf=float(sim_tfidf),
            views_norm=0.0, eng_norm=0.0, rel_norm=0.0, orig_norm=0.0, act_norm=0.0, sim_norm=0.0,
            score=0.0,
            error=""
        )

    except Exception as e:
        return UserScore(
            username=username, eligible=False,
            median_views=0.0, median_engagement=0.0, active_days=0, originality=0.0,
            rel_lex=0.0, sim_tfidf=0.0,
            views_norm=0.0, eng_norm=0.0, rel_norm=0.0, orig_norm=0.0, act_norm=0.0, sim_norm=0.0,
            score=-1e9, error=str(e)
        )


# =========================
# 主流程（模板结构）
# =========================
def main():
    handles = load_handles(INPUT_HANDLES_FILE)
    print(f"Loaded {len(handles)} handles from {INPUT_HANDLES_FILE}")

    # 第一步：对每个用户算特征（过滤在 score_health_kol 里做）
    scores: List[UserScore] = []
    for i, u in enumerate(handles, 1):
        s = score_health_kol(u)
        scores.append(s)

        # 随机抖动，避免被限流
        time.sleep(SLEEP_SEC + random.random() * 0.2)

        if i % 100 == 0:
            print(f"Processed: {i}/{len(handles)}")

    # 第二步：对 eligible 且无错误的用户做归一化
    good_idx = [idx for idx, s in enumerate(scores) if s.eligible and not s.error]

    if not good_idx:
        print("No KOL passed filters.")
        return

    views_list = [scores[idx].median_views for idx in good_idx]
    eng_list = [scores[idx].median_engagement for idx in good_idx]
    rel_list = [scores[idx].rel_lex for idx in good_idx]
    orig_list = [scores[idx].originality for idx in good_idx]
    act_list = [min(scores[idx].active_days / 10.0, 1.0) for idx in good_idx]  # 旧公式里就是 /10 capped
    sim_list = [scores[idx].sim_tfidf for idx in good_idx]

    views_norm = minmax_norm(views_list)
    eng_norm = minmax_norm(eng_list)
    rel_norm = minmax_norm(rel_list)
    orig_norm = minmax_norm(orig_list)
    act_norm = minmax_norm(act_list)
    sim_norm = minmax_norm(sim_list)

    # 第三步：写回 norm + 最终 score
    # 说明：你旧代码的 score 是：
    # 0.35*log1p(median_views) + 0.30*log1p(median_eng) + 0.10*rel + 0.10*originality + 0.15*act
    # 但模板风格通常建议“先归一化再加权”，所以这里改成：
    # 0.35*views_norm + 0.30*eng_norm + 0.10*rel_norm + 0.10*orig_norm + 0.15*act_norm
    # 另外加入一个小权重给 sim_norm（可把它设为 0 关闭）
    W_VIEWS = 0.35
    W_ENG = 0.30
    W_REL = 0.15
    W_ORIG = 0.10
    W_ACT = 0.10
    W_SIM = 0.00  # 想启用 TF-IDF：比如改成 0.05，然后按比例下调其他权重

    for k, idx in enumerate(good_idx):
        s = scores[idx]
        s.views_norm = float(views_norm[k])
        s.eng_norm = float(eng_norm[k])
        s.rel_norm = float(rel_norm[k])
        s.orig_norm = float(orig_norm[k])
        s.act_norm = float(act_norm[k])
        s.sim_norm = float(sim_norm[k])
        s.score = float(
            W_VIEWS * s.views_norm +
            W_ENG * s.eng_norm +
            W_REL * s.rel_norm +
            W_ORIG * s.orig_norm +
            W_ACT * s.act_norm +
            W_SIM * s.sim_norm
        )

    # 第四步：排序（good 在前；其余在后）
    def sort_key(s: UserScore) -> Tuple[int, float]:
        good = (s.eligible and not s.error and s.score > -1e8)
        group = 0 if good else 1
        return (group, -s.score)

    ranked = sorted(scores, key=sort_key)

    # 输出 ranked_full.csv
    with open(OUT_RANKED_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "username", "eligible",
            "median_views", "median_engagement", "active_days", "originality",
            "rel_lex", "sim_tfidf",
            "views_norm", "eng_norm", "rel_norm", "orig_norm", "act_norm", "sim_norm",
            "score", "error"
        ])
        for r, s in enumerate(ranked, 1):
            w.writerow([
                r, s.username, s.eligible,
                f"{s.median_views:.3f}", f"{s.median_engagement:.3f}", s.active_days, f"{s.originality:.4f}",
                f"{s.rel_lex:.6f}", f"{s.sim_tfidf:.6f}",
                f"{s.views_norm:.6f}", f"{s.eng_norm:.6f}", f"{s.rel_norm:.6f}",
                f"{s.orig_norm:.6f}", f"{s.act_norm:.6f}", f"{s.sim_norm:.6f}",
                f"{s.score:.6f}", s.error
            ])

    # 输出 top250.txt（只要 handle）
    top_good = [s for s in ranked if s.eligible and not s.error and s.score > -1e8]
    topN = top_good[:TOP_K]

    with open(OUT_TOP_TXT, "w", encoding="utf-8") as f:
        for s in topN:
            f.write(s.username + "\n")

    print(f"Saved full ranking: {OUT_RANKED_CSV}")
    print(f"Saved top handles (top {TOP_K}): {OUT_TOP_TXT}")
    print(f"Eligible good users: {len(top_good)} / {len(handles)}")

    print("\nTop 10 health KOLs:")
    for s in topN[:10]:
        print(f"{s.username:20s} score={s.score:.4f} median_views={s.median_views:.0f} med_eng={s.median_engagement:.0f} rel={s.rel_lex:.2f} orig={s.originality:.2f} act_days={s.active_days}")


if __name__ == "__main__":
    main()
