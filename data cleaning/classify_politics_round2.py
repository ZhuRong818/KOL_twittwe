#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Set, Tuple, Pattern


# =========================================================
# Existing politics keywords
# =========================================================

POLITICS_PHRASES = {
    "election results",
    "election day",
    "voter turnout",
    "mail in ballot",
    "mail-in ballot",
    "early voting",
    "ballot count",
    "ballot counting",
    "vote count",
    "primary election",
    "general election",
    "presidential election",
    "midterm election",
    "runoff election",
    "campaign trail",
    "presidential debate",
    "candidate forum",
    "concession speech",
    "election night",
    "wins the primary",
    "wins the caucus",
    "wins the seat",
    "sworn into office",
    "senate race",
    "house race",
    "governor race",
    "mayoral race",
    "parliament election",
    "vote to impeach",
    "impeachment vote",
    "supreme court ruling",
    "executive order",
    "passes the bill",
    "signs the bill",
    "government shutdown",
    "cabinet appointment",
    "foreign policy speech",
    "state of the union",
}

POLITICS_ENTITIES = {
    "president",
    "prime minister",
    "senator",
    "senate",
    "congressman",
    "congresswoman",
    "governor",
    "mayor",
    "cabinet secretary",
    "white house",
    "downing street",
    "parliament",
    "congress",
    "house of representatives",
    "election commission",
    "supreme court",
    "mp",
    "mps",
    "minister",
    "candidate",
    "nominee",
}

POLITICS_ACTIONS = {
    "election",
    "elections",
    "vote",
    "votes",
    "voting",
    "ballot",
    "ballots",
    "campaign",
    "campaigning",
    "debate",
    "debates",
    "caucus",
    "referendum",
    "impeach",
    "impeachment",
    "legislation",
    "policy",
    "policies",
    "poll",
    "polls",
    "primary",
    "seat",
    "district",
    "bill",
    "bills",
    "parliamentary",
}

POLITICS_HASHTAGS = {
    "#electionday",
    "#election2024",
    "#election2025",
    "#election2026",
    "#vote2024",
    "#vote2025",
    "#vote2026",
    "#primaryelection",
    "#debatenight",
    "#generalelection",
}


# =========================================================
# Conservative blacklist for noisy mined terms
# =========================================================

NOISY_UNIGRAMS = {
    "movie", "film", "album", "song", "trailer", "netflix", "spotify",
    "earnings", "premiere", "concert", "tour", "show", "episode",
}

NOISY_BIGRAMS = {
    "box office", "movie trailer", "album release", "concert tour",
    "streaming release", "earnings report",
}

NOISY_HASHTAGS = {
    "#earnings", "#nowplaying", "#albumrelease", "#netflix", "#disney",
}


# =========================================================
# Field candidates
# =========================================================

TEXT_FIELDS_CANDIDATES = [
    "text",
    "full_text",
    "content",
    "tweet_text",
    "rawContent",
    "renderedContent",
]

USERNAME_FIELDS_CANDIDATES = [
    "username",
    "userName",
    "screen_name",
    "author_username",
    "authorUsername",
    "handle",
]

AUTHOR_OBJ_FIELDS = [
    "author",
    "user",
    "account",
]

URL_RE = re.compile(r"https?://\S+", flags=re.IGNORECASE)
HASHTAG_RE = re.compile(r"#\w+")
NON_WORD_RE = re.compile(r"[^\w\s#\$-]+")


# =========================================================
# Scoring
# =========================================================

SCORE_EXISTING_PHRASE = 2.5
SCORE_EXISTING_ENTITY = 1.0
SCORE_EXISTING_ACTION = 1.0
SCORE_EXISTING_HASHTAG = 1.5

SCORE_MINED_BIGRAM = 3.0
SCORE_MINED_UNIGRAM = 1.0
SCORE_MINED_HASHTAG = 2.0

SCORE_KOL_PRIOR = 1.0
SCORE_COOCCUR_BOOST = 1.5

MIN_SCORE_POLITICS = 2.5


# =========================================================
# Helpers
# =========================================================

def normalize_text(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = text.replace("/", " ")
    text = NON_WORD_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_username(u: str) -> str:
    u = (u or "").strip().lower()
    if u.startswith("@"):
        u = u[1:]
    return u

def extract_hashtags(text: str) -> Set[str]:
    return {m.group(0).lower() for m in HASHTAG_RE.finditer(text or "")}

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
        ["tweet", "text"],
        ["legacy", "full_text"],
        ["data", "text"],
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

def get_username_from_record(obj: Dict[str, Any]) -> str:
    for key in USERNAME_FIELDS_CANDIDATES:
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return normalize_username(v)

    for author_key in AUTHOR_OBJ_FIELDS:
        av = obj.get(author_key)
        if isinstance(av, dict):
            for key in USERNAME_FIELDS_CANDIDATES:
                v = av.get(key)
                if isinstance(v, str) and v.strip():
                    return normalize_username(v)

    nested_candidates = [
        ["tweet", "author", "userName"],
        ["legacy", "screen_name"],
        ["author", "legacy", "screen_name"],
    ]
    for path in nested_candidates:
        v = deep_get(obj, path)
        if isinstance(v, str) and v.strip():
            return normalize_username(v)

    return ""

def load_handle_file(path: str) -> Set[str]:
    handles: Set[str] = set()
    if not os.path.exists(path):
        return handles
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            h = normalize_username(line.strip())
            if h:
                handles.add(h)
    return handles


# =========================================================
# Regex compilers
# =========================================================

def build_phrase_pattern(phrases: Set[str]) -> Optional[Pattern[str]]:
    if not phrases:
        return None
    escaped = sorted((re.escape(p) for p in phrases), key=len, reverse=True)
    return re.compile(r"(?:%s)" % "|".join(escaped))

def build_token_pattern(tokens: Set[str]) -> Optional[Pattern[str]]:
    if not tokens:
        return None
    escaped = sorted((re.escape(t) for t in tokens), key=len, reverse=True)
    return re.compile(r"(?<!\w)(%s)(?!\w)" % "|".join(escaped))

def contains_phrase_compiled(text: str, pattern: Optional[Pattern[str]]) -> Tuple[bool, List[str]]:
    if pattern is None:
        return False, []
    matches = sorted(set(m.lower() for m in pattern.findall(text)))
    return (len(matches) > 0, matches)

def contains_token_compiled(text: str, pattern: Optional[Pattern[str]]) -> Tuple[bool, List[str]]:
    if pattern is None:
        return False, []
    matches = sorted(set(m.lower() for m in pattern.findall(text)))
    return (len(matches) > 0, matches)


# =========================================================
# Load mined CSV keywords
# =========================================================

def parse_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default

def parse_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except Exception:
        return default

def load_mined_keywords(
    unigram_csv: str,
    bigram_csv: str,
    hashtag_csv: str,
    min_unigram_pos_df: int = 50,
    min_unigram_log_odds: float = 2.0,
    min_bigram_pos_df: int = 20,
    min_bigram_log_odds: float = 3.0,
    min_hashtag_pos_df: int = 5,
    min_hashtag_log_odds: float = 2.0,
) -> Tuple[Set[str], Set[str], Set[str]]:
    mined_unigrams: Set[str] = set()
    mined_bigrams: Set[str] = set()
    mined_hashtags: Set[str] = set()

    if os.path.exists(unigram_csv):
        with open(unigram_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                feat = (row.get("feature") or "").strip().lower()
                if not feat or feat in NOISY_UNIGRAMS or " " in feat:
                    continue
                pos_df = parse_int(row, "pos_df")
                log_odds = parse_float(row, "log_odds")
                if pos_df >= min_unigram_pos_df and log_odds >= min_unigram_log_odds:
                    mined_unigrams.add(feat)

    if os.path.exists(bigram_csv):
        with open(bigram_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                feat = (row.get("feature") or "").strip().lower()
                if not feat or feat in NOISY_BIGRAMS or " " not in feat:
                    continue
                pos_df = parse_int(row, "pos_df")
                log_odds = parse_float(row, "log_odds")
                if pos_df >= min_bigram_pos_df and log_odds >= min_bigram_log_odds:
                    mined_bigrams.add(feat)

    if os.path.exists(hashtag_csv):
        with open(hashtag_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                feat = (row.get("feature") or "").strip().lower()
                if not feat or feat in NOISY_HASHTAGS or not feat.startswith("#"):
                    continue
                pos_df = parse_int(row, "pos_df")
                log_odds = parse_float(row, "log_odds")
                if pos_df >= min_hashtag_pos_df and log_odds >= min_hashtag_log_odds:
                    mined_hashtags.add(feat)

    return mined_unigrams, mined_bigrams, mined_hashtags


# =========================================================
# Politics classifier
# =========================================================

class PoliticsClassifier:
    def __init__(
        self,
        politics_kol_handles: Set[str],
        mined_unigrams: Set[str],
        mined_bigrams: Set[str],
        mined_hashtags: Set[str],
    ) -> None:
        self.politics_kol_handles = politics_kol_handles

        self.existing_phrase_re = build_phrase_pattern(POLITICS_PHRASES)
        self.existing_entity_re = build_token_pattern(POLITICS_ENTITIES)
        self.existing_action_re = build_token_pattern(POLITICS_ACTIONS)

        self.mined_unigram_re = build_token_pattern(mined_unigrams)
        self.mined_bigram_re = build_phrase_pattern(mined_bigrams)

        self.mined_hashtags = mined_hashtags

    def classify(self, raw_text: str, username: str) -> Tuple[bool, Dict[str, Any]]:
        text_norm = normalize_text(raw_text)
        hashtags = extract_hashtags(raw_text)

        score = 0.0

        matched_existing_phrases: List[str] = []
        matched_existing_entities: List[str] = []
        matched_existing_actions: List[str] = []
        matched_existing_hashtags: List[str] = []

        matched_mined_bigrams: List[str] = []
        matched_mined_unigrams: List[str] = []
        matched_mined_hashtags: List[str] = []

        author_prior_hit = False
        cooccur_signals: List[str] = []

        # Existing rules
        hit, vals = contains_phrase_compiled(text_norm, self.existing_phrase_re)
        if hit:
            score += SCORE_EXISTING_PHRASE
            matched_existing_phrases.extend(vals)

        hit, vals = contains_token_compiled(text_norm, self.existing_entity_re)
        if hit:
            score += SCORE_EXISTING_ENTITY
            matched_existing_entities.extend(vals)

        hit, vals = contains_token_compiled(text_norm, self.existing_action_re)
        if hit:
            score += SCORE_EXISTING_ACTION
            matched_existing_actions.extend(vals)

        existing_hash = sorted(list(hashtags & POLITICS_HASHTAGS))
        if existing_hash:
            score += SCORE_EXISTING_HASHTAG
            matched_existing_hashtags.extend(existing_hash)

        # Mined keywords
        hit, vals = contains_phrase_compiled(text_norm, self.mined_bigram_re)
        if hit:
            score += SCORE_MINED_BIGRAM
            matched_mined_bigrams.extend(vals)

        hit, vals = contains_token_compiled(text_norm, self.mined_unigram_re)
        if hit:
            score += SCORE_MINED_UNIGRAM
            matched_mined_unigrams.extend(vals)

        mined_hash = sorted(list(hashtags & self.mined_hashtags))
        if mined_hash:
            score += SCORE_MINED_HASHTAG
            matched_mined_hashtags.extend(mined_hash)

        # KOL prior
        if username and username in self.politics_kol_handles:
            score += SCORE_KOL_PRIOR
            author_prior_hit = True

        # Co-occurrence boosts
        if "senate" in text_norm and "race" in text_norm:
            score += SCORE_COOCCUR_BOOST
            cooccur_signals.append("senate+race")

        if "vote" in text_norm and "district" in text_norm:
            score += SCORE_COOCCUR_BOOST
            cooccur_signals.append("vote+district")

        if "ballot" in text_norm and "count" in text_norm:
            score += SCORE_COOCCUR_BOOST
            cooccur_signals.append("ballot+count")

        if "supreme court" in text_norm and ("appeal" in text_norm or "ruling" in text_norm or "order" in text_norm):
            score += SCORE_COOCCUR_BOOST
            cooccur_signals.append("supreme_court+legal_action")

        if "executive" in text_norm and "order" in text_norm:
            score += SCORE_COOCCUR_BOOST
            cooccur_signals.append("executive+order")

        matched = score >= MIN_SCORE_POLITICS

        details = {
            "score": round(score, 3),
            "matched_existing_phrases": sorted(set(matched_existing_phrases)),
            "matched_existing_entities": sorted(set(matched_existing_entities)),
            "matched_existing_actions": sorted(set(matched_existing_actions)),
            "matched_existing_hashtags": sorted(set(matched_existing_hashtags)),
            "matched_mined_bigrams": sorted(set(matched_mined_bigrams)),
            "matched_mined_unigrams": sorted(set(matched_mined_unigrams)),
            "matched_mined_hashtags": sorted(set(matched_mined_hashtags)),
            "author_prior_hit": author_prior_hit,
            "cooccur_signals": sorted(set(cooccur_signals)),
        }
        return matched, details


# =========================================================
# Main processing
# =========================================================

def process_file(
    input_path: str,
    output_politics_path: str,
    output_remaining_path: str,
    backup: bool,
    prior_dir: str,
    unigram_csv: str,
    bigram_csv: str,
    hashtag_csv: str,
    add_audit_fields: bool,
) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    politics_kol_handles = load_handle_file(os.path.join(prior_dir, "politics_top_242.txt"))

    mined_unigrams, mined_bigrams, mined_hashtags = load_mined_keywords(
        unigram_csv=unigram_csv,
        bigram_csv=bigram_csv,
        hashtag_csv=hashtag_csv,
    )

    print("[INFO] Loaded politics KOL handles:", len(politics_kol_handles))
    print("[INFO] Loaded mined unigrams     :", len(mined_unigrams))
    print("[INFO] Loaded mined bigrams      :", len(mined_bigrams))
    print("[INFO] Loaded mined hashtags     :", len(mined_hashtags))

    classifier = PoliticsClassifier(
        politics_kol_handles=politics_kol_handles,
        mined_unigrams=mined_unigrams,
        mined_bigrams=mined_bigrams,
        mined_hashtags=mined_hashtags,
    )

    if backup:
        backup_path = input_path + ".bak"
        shutil.copy2(input_path, backup_path)
        print(f"[INFO] Backup created: {backup_path}")

    total = 0
    matched_count = 0
    no_text_count = 0
    json_error_count = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_politics_path, "w", encoding="utf-8") as fout_pol, \
         open(output_remaining_path, "w", encoding="utf-8") as fout_rem:

        for line in fin:
            raw_line = line.rstrip("\n")
            if not raw_line.strip():
                continue

            total += 1

            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                fout_rem.write(raw_line + "\n")
                json_error_count += 1
                continue

            raw_text = get_text_from_record(obj)
            username = get_username_from_record(obj)

            if not raw_text:
                fout_rem.write(raw_line + "\n")
                no_text_count += 1
                continue

            is_politics, details = classifier.classify(raw_text, username)

            if is_politics:
                if add_audit_fields:
                    obj["_matched_categories"] = ["elections_politics"]
                    obj["_category_match_details"] = {"elections_politics": details}
                    obj["_author_username_normalized"] = username
                    fout_pol.write(json.dumps(obj, ensure_ascii=False) + "\n")
                else:
                    fout_pol.write(raw_line + "\n")
                matched_count += 1
            else:
                fout_rem.write(raw_line + "\n")

            if total % 100000 == 0:
                print(
                    f"[INFO] processed={total}, matched_politics={matched_count}, "
                    f"remaining={total - matched_count}"
                )

    print("\n[DONE]")
    print(f"Total processed        : {total}")
    print(f"Matched politics       : {matched_count}")
    print(f"Remaining              : {total - matched_count}")
    print(f"No text count          : {no_text_count}")
    print(f"JSON error count       : {json_error_count}")
    print(f"Politics output file   : {output_politics_path}")
    print(f"Remaining output file  : {output_remaining_path}")


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to remaining JSONL to classify")
    parser.add_argument("--output-politics", type=str, default="elections_politics_round2.jsonl")
    parser.add_argument("--output-remaining", type=str, default="remaining_after_politics_round2.jsonl")
    parser.add_argument("--prior-dir", type=str, default=".", help="Directory containing politics_top_242.txt")
    parser.add_argument("--unigram-csv", type=str, default="/mnt/data/politics_unigrams.csv")
    parser.add_argument("--bigram-csv", type=str, default="/mnt/data/politics_bigrams.csv")
    parser.add_argument("--hashtag-csv", type=str, default="/mnt/data/politics_hashtags.csv")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--add-audit-fields", action="store_true", help="Add matched details into matched output")
    return parser.parse_args()

def main():
    args = parse_args()
    process_file(
        input_path=args.input,
        output_politics_path=args.output_politics,
        output_remaining_path=args.output_remaining,
        backup=not args.no_backup,
        prior_dir=args.prior_dir,
        unigram_csv=args.unigram_csv,
        bigram_csv=args.bigram_csv,
        hashtag_csv=args.hashtag_csv,
        add_audit_fields=args.add_audit_fields,
    )

if __name__ == "__main__":
    main()