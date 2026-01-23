import csv
import json
import math
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 配置区（你只需要改这里）
# =========================
API_KEY = "new1_c64438393d9444c18cb94828b9511f27"
BASE = "https://api.twitterapi.io"

INPUT_HANDLES_FILE = "handles.txt"

# 你要求：每个账号只拉 10 条推文
TWEETS_PER_USER = 10

# 你要求：粉丝数 >= 20万
MIN_FOLLOWERS = 200_000

# 你要求：权重 0.4 + 0.6
W_SIM = 0.4
W_ENG = 0.6

# 输出文件
OUT_RANKED_CSV = "ranked_full.csv"
OUT_TOP300_TXT = "top300.txt"

# 请求节流：避免 429
SLEEP_SEC = 0.20
RETRY_429_SLEEP_SEC = 5.0
TIMEOUT_SEC = 30


# =========================
# 金融主题词表（通用）
# =========================
FIN_KEYPHRASES = [
    # 股票/基本面
    "stocks", "equities", "earnings", "revenue", "profit", "guidance", "valuation",
    "price to earnings", "p/e", "free cash flow", "fcf", "cash flow", "balance sheet",
    "dividend", "buyback", "margin", "growth", "fundamentals",

    # 宏观/利率
    "inflation", "interest rates", "rate hike", "rate cut", "federal reserve", "fed",
    "central bank", "treasury yields", "bond yields", "yield curve", "recession",
    "gdp", "unemployment", "macro", "soft landing", "hard landing",

    # 市场/风险
    "market", "risk", "volatility", "drawdown", "liquidity", "bull market", "bear market",
    "asset allocation", "portfolio", "hedge", "correlation",

    # ETF/基金/策略
    "etf", "index fund", "mutual fund", "factor investing", "value investing",
    "growth investing", "quant", "systematic", "backtest",

    # 衍生品
    "options", "calls", "puts", "implied volatility", "gamma", "delta", "theta",

    # 加密
    "crypto", "bitcoin", "btc", "ethereum", "eth", "on-chain", "stablecoin"
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

    sim_raw: float          # 0~1（cosine max then avg）
    eng_raw: float          # log 压缩后的 raw engagement score（未归一化）

    sim_norm: float         # min-max 后 0~1
    eng_norm: float         # min-max 后 0~1
    score: float            # 最终 score

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
    for _ in range(3):
        r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT_SEC)
        if r.status_code == 429:
            time.sleep(RETRY_429_SLEEP_SEC)
            continue
        r.raise_for_status()
        return r.json()
    # 最后一次
    r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    return r.json()


def fetch_user_info(username: str) -> UserProfile:
    data = safe_get(f"{BASE}/twitter/user/info", params={"userName": username})
    # 常见结构：{"status":"success","data":{...}}
    d = data.get("data", data)
    followers = int(d.get("followers", 0) or 0)
    return UserProfile(username=username, followers=followers)


def fetch_last_tweets(username: str, limit: int = 10, include_replies: bool = False) -> List[Tweet]:
    data = safe_get(
        f"{BASE}/twitter/user/last_tweets",
        params={"userName": username, "includeReplies": str(include_replies).lower(), "cursor": ""}
    )
    # 常见结构：{"tweets":[{...},...], "nextCursor": "..."}
    tweets_raw = data.get("tweets", []) or []
    tweets_raw = tweets_raw[:limit]

    out = []
    for t in tweets_raw:
        out.append(
            Tweet(
                text=(t.get("text") or "").strip(),
                likeCount=int(t.get("likeCount") or 0),
                replyCount=int(t.get("replyCount") or 0),
                retweetCount=int(t.get("retweetCount") or 0),
                quoteCount=int(t.get("quoteCount") or 0),
            )
        )
    return out


def compute_similarity_avg(texts: List[str], keyphrases: List[str]) -> float:
    """
    用 TF-IDF + cosine similarity：
    每条推文对 keyphrases 求 cosine，相似度取 max；然后对所有推文取平均。
    输出范围大致在 [0, 1]。
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
    # 每条推文取和金融主题最接近的那个短语
    max_per_tweet = sims.max(axis=1)
    return float(max_per_tweet.mean())


def compute_engagement_raw(tweets: List[Tweet], followers: int) -> float:
    """
    engagement = (like+reply+retweet+quote) / followers
    然后做 log1p 压缩，避免极端值
    """
    if not tweets:
        return 0.0
    f = max(followers, 1)
    ers = []
    for t in tweets:
        inter = t.likeCount + t.replyCount + t.retweetCount + t.quoteCount
        ers.append(inter / f)
    er_avg = sum(ers) / len(ers)
    return float(math.log1p(er_avg * 1000.0))


def minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def main():
    handles = load_handles(INPUT_HANDLES_FILE)
    print(f"Loaded {len(handles)} handles from {INPUT_HANDLES_FILE}")

    # 第一步：拉 profile（粉丝数）
    profiles: Dict[str, UserProfile] = {}
    for i, u in enumerate(handles, 1):
        try:
            profiles[u] = fetch_user_info(u)
        except Exception as e:
            # profile 拉不到也记录
            profiles[u] = UserProfile(username=u, followers=0)
            print(f"[WARN] user/info failed for {u}: {e}")
        time.sleep(SLEEP_SEC)
        if i % 100 == 0:
            print(f"Fetched profiles: {i}/{len(handles)}")

    # 第二步：只对 eligible 的账号拉推文 + 算 raw 分
    temp_scores: List[UserScore] = []
    for i, u in enumerate(handles, 1):
        p = profiles[u]
        eligible = p.followers >= MIN_FOLLOWERS

        if not eligible:
            temp_scores.append(UserScore(
                username=u,
                followers=p.followers,
                eligible=False,
                sim_raw=0.0,
                eng_raw=0.0,
                sim_norm=0.0,
                eng_norm=0.0,
                score=-1e9,
                error="followers_below_threshold"
            ))
            continue

        try:
            tweets = fetch_last_tweets(u, limit=TWEETS_PER_USER, include_replies=False)
            texts = [t.text for t in tweets]

            sim_raw = compute_similarity_avg(texts, FIN_KEYPHRASES)
            eng_raw = compute_engagement_raw(tweets, p.followers)

            temp_scores.append(UserScore(
                username=u,
                followers=p.followers,
                eligible=True,
                sim_raw=sim_raw,
                eng_raw=eng_raw,
                sim_norm=0.0,
                eng_norm=0.0,
                score=0.0,
                error=""
            ))
        except Exception as e:
            temp_scores.append(UserScore(
                username=u,
                followers=p.followers,
                eligible=True,
                sim_raw=0.0,
                eng_raw=0.0,
                sim_norm=0.0,
                eng_norm=0.0,
                score=-1e9,
                error=str(e)
            ))

        time.sleep(SLEEP_SEC)
        if i % 100 == 0:
            print(f"Scored users: {i}/{len(handles)}")

    # 第三步：对 eligible 用户做归一化（让 0.5/0.5 权重有意义）
    eligible_indices = [idx for idx, s in enumerate(temp_scores) if s.eligible and s.score > -1e8 and not s.error]
    sim_list = [temp_scores[idx].sim_raw for idx in eligible_indices]
    eng_list = [temp_scores[idx].eng_raw for idx in eligible_indices]

    sim_norm_list = minmax_norm(sim_list)
    eng_norm_list = minmax_norm(eng_list)

    # 写回 norm + 最终 score
    for k, idx in enumerate(eligible_indices):
        s = temp_scores[idx]
        s.sim_norm = float(sim_norm_list[k])
        s.eng_norm = float(eng_norm_list[k])
        s.score = float(W_SIM * s.sim_norm + W_ENG * s.eng_norm)

    # 第四步：排序（eligible 且无错误的在前；其他在后）
    def sort_key(s: UserScore) -> Tuple[int, float]:
        # eligible & no error: group 0; else group 1
        good = (s.eligible and not s.error and s.score > -1e8)
        group = 0 if good else 1
        return (group, -s.score)

    ranked = sorted(temp_scores, key=sort_key)

    # 输出 ranked_full.csv（从上到下完整名单）
    with open(OUT_RANKED_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "username", "followers", "eligible",
            "sim_raw", "eng_raw", "sim_norm", "eng_norm", "score", "error"
        ])
        for r, s in enumerate(ranked, 1):
            w.writerow([
                r, s.username, s.followers, s.eligible,
                f"{s.sim_raw:.6f}", f"{s.eng_raw:.6f}",
                f"{s.sim_norm:.6f}", f"{s.eng_norm:.6f}",
                f"{s.score:.6f}", s.error
            ])

    # 输出 top300.txt（只要 handle）
    # 只取 “eligible 且无错误” 的 top300
    top_good = [s for s in ranked if s.eligible and not s.error and s.score > -1e8]
    top300 = top_good[:300]

    with open(OUT_TOP300_TXT, "w", encoding="utf-8") as f:
        for s in top300:
            f.write(s.username + "\n")

    print(f"Saved full ranking: {OUT_RANKED_CSV}")
    print(f"Saved top300 handles: {OUT_TOP300_TXT}")
    print(f"Eligible good users: {len(top_good)} / {len(handles)}")


if __name__ == "__main__":
    main()
