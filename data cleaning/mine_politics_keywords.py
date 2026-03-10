#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


TEXT_FIELDS_CANDIDATES = [
    "text",
    "full_text",
    "content",
    "tweet_text",
    "rawContent",
    "renderedContent",
]

URL_RE = re.compile(r"https?://\S+", flags=re.IGNORECASE)
HASHTAG_RE = re.compile(r"#\w+")
NON_WORD_RE = re.compile(r"[^\w\s#]+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "i", "if", "in", "is",
    "it", "its", "me", "my", "of", "on", "or", "our", "so", "that", "the",
    "their", "them", "they", "this", "to", "us", "was", "we", "were", "will",
    "with", "you", "your", "yours", "rt", "via", "amp"
}

MIN_TOKEN_LEN = 3


def deep_get(d: Any, path: List[str]) -> Optional[Any]:
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def get_text_from_record(obj: Dict[str, Any]) -> str:
    parts: List[str] = []

    for key in TEXT_FIELDS_CANDIDATES:
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())

    for key in ("ocr_text", "image_text", "alt_text"):
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())

    nested_candidates = [
        ["legacy", "full_text"],
        ["data", "text"],
        ["tweet", "text"],
        ["note_tweet", "note_tweet_results", "result", "text"],
    ]
    for path in nested_candidates:
        v = deep_get(obj, path)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())

    seen = set()
    deduped = []
    for p in parts:
        if p not in seen:
            deduped.append(p)
            seen.add(p)

    return " ".join(deduped).strip()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = text.replace("/", " ")
    text = NON_WORD_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_hashtags(text: str) -> Set[str]:
    return {m.group(0).lower() for m in HASHTAG_RE.finditer(text or "")}


def tokenize(text: str) -> List[str]:
    toks = text.split()
    out = []
    for tok in toks:
        if tok.startswith("#"):
            continue
        if len(tok) < MIN_TOKEN_LEN:
            continue
        if tok.isdigit():
            continue
        if tok in STOPWORDS:
            continue
        out.append(tok)
    return out


def make_bigrams(tokens: List[str]) -> List[str]:
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]


def update_document_frequency(
    counter: Counter,
    items: Iterable[str],
) -> None:
    for x in set(items):
        counter[x] += 1


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def collect_feature_dfs(
    path: str,
    max_docs: Optional[int] = None,
) -> Tuple[int, Counter, Counter, Counter]:
    """
    Returns:
        num_docs,
        unigram_df,
        bigram_df,
        hashtag_df
    """
    unigram_df = Counter()
    bigram_df = Counter()
    hashtag_df = Counter()

    num_docs = 0

    for obj in iter_jsonl(path):
        text_raw = get_text_from_record(obj)
        if not text_raw:
            continue

        num_docs += 1
        if max_docs is not None and num_docs > max_docs:
            break

        hashtags = extract_hashtags(text_raw)
        text_norm = normalize_text(text_raw)
        tokens = tokenize(text_norm)
        bigrams = make_bigrams(tokens)

        update_document_frequency(unigram_df, tokens)
        update_document_frequency(bigram_df, bigrams)
        update_document_frequency(hashtag_df, hashtags)

    return num_docs, unigram_df, bigram_df, hashtag_df


def score_features(
    pos_docs: int,
    bg_docs: int,
    pos_df: Counter,
    bg_df: Counter,
    min_pos_df: int,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Compute distinctiveness scores for each feature.
    """
    features = set(pos_df.keys())
    rows: List[Dict[str, Any]] = []

    for feat in features:
        p = pos_df[feat]
        b = bg_df.get(feat, 0)

        if p < min_pos_df:
            continue

        pos_rate = p / pos_docs if pos_docs else 0.0
        bg_rate = b / bg_docs if bg_docs else 0.0

        # smoothing
        lift = (pos_rate + 1e-9) / (bg_rate + 1e-9)

        # smoothed log-odds
        log_odds = math.log((p + 0.5) / (pos_docs - p + 0.5)) - math.log((b + 0.5) / (bg_docs - b + 0.5))

        rows.append({
            "feature": feat,
            "pos_df": p,
            "bg_df": b,
            "pos_rate": round(pos_rate, 8),
            "bg_rate": round(bg_rate, 8),
            "lift": round(lift, 4),
            "log_odds": round(log_odds, 4),
        })

    rows.sort(key=lambda x: (x["log_odds"], x["pos_df"]), reverse=True)
    return rows[:top_k]


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive", required=True, help="Path to elections_politics.jsonl")
    parser.add_argument("--background", required=True, help="Path to remaining merged_all.jsonl or other background pool")
    parser.add_argument("--outdir", default="politics_keyword_mining", help="Output directory")
    parser.add_argument("--max-positive", type=int, default=None, help="Optional cap on number of positive docs")
    parser.add_argument("--max-background", type=int, default=None, help="Optional cap on number of background docs")
    parser.add_argument("--min-pos-df-word", type=int, default=20)
    parser.add_argument("--min-pos-df-bigram", type=int, default=10)
    parser.add_argument("--min-pos-df-hashtag", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("[INFO] Collecting positive feature document frequencies...")
    pos_docs, pos_uni, pos_bi, pos_hash = collect_feature_dfs(
        args.positive,
        max_docs=args.max_positive,
    )
    print(f"[INFO] Positive docs: {pos_docs:,}")

    print("[INFO] Collecting background feature document frequencies...")
    bg_docs, bg_uni, bg_bi, bg_hash = collect_feature_dfs(
        args.background,
        max_docs=args.max_background,
    )
    print(f"[INFO] Background docs: {bg_docs:,}")

    print("[INFO] Scoring unigrams...")
    unigram_rows = score_features(
        pos_docs=pos_docs,
        bg_docs=bg_docs,
        pos_df=pos_uni,
        bg_df=bg_uni,
        min_pos_df=args.min_pos_df_word,
        top_k=args.top_k,
    )

    print("[INFO] Scoring bigrams...")
    bigram_rows = score_features(
        pos_docs=pos_docs,
        bg_docs=bg_docs,
        pos_df=pos_bi,
        bg_df=bg_bi,
        min_pos_df=args.min_pos_df_bigram,
        top_k=args.top_k,
    )

    print("[INFO] Scoring hashtags...")
    hashtag_rows = score_features(
        pos_docs=pos_docs,
        bg_docs=bg_docs,
        pos_df=pos_hash,
        bg_df=bg_hash,
        min_pos_df=args.min_pos_df_hashtag,
        top_k=args.top_k,
    )

    write_csv(os.path.join(args.outdir, "politics_unigrams.csv"), unigram_rows)
    write_csv(os.path.join(args.outdir, "politics_bigrams.csv"), bigram_rows)
    write_csv(os.path.join(args.outdir, "politics_hashtags.csv"), hashtag_rows)

    print("[DONE]")
    print(f"[INFO] Wrote: {os.path.join(args.outdir, 'politics_unigrams.csv')}")
    print(f"[INFO] Wrote: {os.path.join(args.outdir, 'politics_bigrams.csv')}")
    print(f"[INFO] Wrote: {os.path.join(args.outdir, 'politics_hashtags.csv')}")


if __name__ == "__main__":
    main()