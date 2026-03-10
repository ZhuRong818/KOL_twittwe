#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import multiprocessing as mp
from itertools import islice
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional, Pattern


# =========================
# Dictionaries
# =========================

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

SPORTS_PHRASES = {
    "final score",
    "full time",
    "extra time",
    "penalty shootout",
    "overtime winner",
    "walk off home run",
    "walk-off home run",
    "match point",
    "wins game 7",
    "game 7",
    "series tied",
    "playoff berth",
    "playoff game",
    "season opener",
    "season finale",
    "transfer window",
    "trade deadline",
    "injury report",
    "starting lineup",
    "starting quarterback",
    "starting pitcher",
    "red card",
    "yellow card",
    "hat trick",
    "grand slam title",
    "knockout stage",
    "group stage",
    "world cup qualifier",
    "champions league",
    "nba finals",
    "nfl draft",
    "super bowl",
    "stanley cup",
    "world series",
    "atp finals",
    "wta finals",
    "olympic final",
}

SPORTS_ENTITIES = {
    "nba",
    "nfl",
    "mlb",
    "nhl",
    "ufc",
    "fifa",
    "uefa",
    "atp",
    "wta",
    "formula 1",
    "premier league",
    "champions league",
    "la liga",
    "serie a",
    "bundesliga",
    "world cup",
    "olympics",
    "wimbledon",
    "us open",
    "australian open",
    "roland garros",
    "super bowl",
    "world series",
    "stanley cup",
    "grand slam",
    "playoffs",
    "playoff",
}

SPORTS_ACTIONS = {
    "playoff",
    "playoffs",
    "tournament",
    "championship",
    "season",
    "draft",
    "match",
    "game",
    "score",
    "scored",
    "goal",
    "goals",
    "won",
    "wins",
    "beat",
    "beats",
    "defeat",
    "defeats",
    "defeated",
    "final",
    "finals",
    "semifinal",
    "quarterfinal",
    "overtime",
    "penalty",
    "injury",
    "transfer",
    "trade",
}

SPORTS_HASHTAGS = {
    "#nbafinals",
    "#superbowl",
    "#worldcup",
    "#championsleague",
    "#nfldraft",
    "#wimbledon",
    "#olympics",
    "#nba",
    "#nfl",
    "#mlb",
    "#nhl",
}

ENT_PHRASES = {
    "academy awards",
    "oscar nomination",
    "oscar winner",
    "grammy nomination",
    "grammy winner",
    "emmy nomination",
    "emmy winner",
    "golden globe winner",
    "bafta winner",
    "best picture",
    "best actor",
    "best actress",
    "album of the year",
    "song of the year",
    "record of the year",
    "red carpet appearance",
    "award show",
    "acceptance speech",
    "nominee list",
    "nominated for",
    "wins best picture",
    "wins best actor",
    "wins best actress",
    "box office debut",
    "opening weekend",
    "season premiere",
    "series finale",
    "film festival premiere",
    "new album",
    "album release",
    "tour dates",
    "sold out show",
    "sold-out show",
    "concert tour",
    "movie trailer",
    "official trailer",
    "streaming release",
    "box office",
    "soundtrack release",
}

ENT_ENTITIES = {
    "oscars",
    "oscar",
    "grammys",
    "grammy",
    "emmys",
    "emmy",
    "golden globes",
    "golden globe",
    "bafta",
    "cannes",
    "sundance",
    "billboard music awards",
    "mtv video music awards",
    "tony awards",
    "met gala",
    "netflix",
    "disney",
    "hbo",
    "spotify",
    "album",
    "movie",
    "film",
    "series",
}

ENT_ACTIONS = {
    "album",
    "film",
    "movie",
    "actor",
    "actress",
    "concert",
    "tour",
    "premiere",
    "trailer",
    "soundtrack",
    "box office",
    "red carpet",
    "performance",
    "award",
    "awards",
    "nominee",
    "nomination",
    "winner",
    "winners",
    "streaming",
    "episode",
    "episodes",
    "season",
}

ENT_HASHTAGS = {
    "#oscars",
    "#grammys",
    "#emmys",
    "#goldenglobes",
    "#redcarpet",
    "#nowplaying",
    "#albumrelease",
    "#netflix",
    "#disney",
}

COMPANY_PHRASES = {
    "product launch",
    "official launch",
    "launch event",
    "announced today",
    "introduces new",
    "unveils new",
    "rolls out",
    "rolling out",
    "beta release",
    "public beta",
    "feature update",
    "software update",
    "hardware update",
    "press release",
    "earnings report",
    "quarterly results",
    "raises guidance",
    "cuts guidance",
    "reports revenue",
    "investor day",
    "developer conference",
    "keynote event",
    "partnership announcement",
    "acquisition announcement",
    "merger announcement",
    "ipo filing",
    "sec filing",
    "roadmap update",
    "general availability",
    "api release",
    "model release",
    "new device",
    "new product line",
}

COMPANY_ENTITIES = {
    "apple",
    "google",
    "microsoft",
    "meta",
    "amazon",
    "tesla",
    "nvidia",
    "openai",
    "samsung",
    "sony",
    "intel",
    "amd",
    "netflix",
    "disney",
    "adobe",
    "salesforce",
    "oracle",
    "uber",
    "airbnb",
    "spotify",
    "iphone",
    "pixel",
    "galaxy",
    "macbook",
    "windows",
    "chatgpt",
    "copilot",
    "azure",
    "aws",
    "playstation",
    "xbox",
    "gemini",
    "claude",
    "ios",
    "android",
    "app",
    "api",
}

COMPANY_ACTIONS = {
    "launch",
    "launches",
    "launched",
    "unveil",
    "unveils",
    "unveiled",
    "announce",
    "announces",
    "announced",
    "release",
    "releases",
    "released",
    "update",
    "updates",
    "updated",
    "earnings",
    "keynote",
    "partnership",
    "acquisition",
    "merger",
    "ipo",
    "filing",
    "api",
    "product",
    "feature",
    "roadmap",
    "rollout",
    "beta",
    "preview",
    "availability",
    "revenue",
    "guidance",
    "conference",
    "model",
    "device",
}

COMPANY_HASHTAGS = {
    "#productlaunch",
    "#earnings",
    "#keynote",
    "#nowavailable",
    "#pressrelease",
    "#launchevent",
    "#openai",
    "#apple",
    "#google",
    "#microsoft",
    "#nvidia",
}

COMPANY_CASHTAGS = {
    "$aapl", "$msft", "$googl", "$goog", "$meta", "$amzn",
    "$tsla", "$nvda", "$amd", "$nflx", "$dis", "$intc"
}

CATEGORY_TO_FILE = {
    "elections_politics": "elections_politics.jsonl",
    "sports": "sports.jsonl",
    "entertainment": "entertainment.jsonl",
    "company_product_announcements": "company_product_announcements.jsonl",
}

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
CASHTAG_RE = re.compile(r"\$\w+")
NON_WORD_RE = re.compile(r"[^\w\s#\$-]+")

SCORE_PHRASE = 2.0
SCORE_ENTITY = 1.0
SCORE_ACTION = 1.0
SCORE_HASHTAG = 1.5
SCORE_CASHTAG = 1.5

KOL_PRIOR_BOOST_STRONG = 1.0
KOL_PRIOR_BOOST_WEAK = 0.5

MIN_CATEGORY_SCORE = 1.5


# =========================
# Precompiled patterns
# =========================

def build_phrase_pattern(phrases: Set[str]) -> Pattern[str]:
    escaped = sorted((re.escape(p) for p in phrases), key=len, reverse=True)
    return re.compile(r"(?:%s)" % "|".join(escaped))

def build_token_pattern(tokens: Set[str]) -> Pattern[str]:
    escaped = sorted((re.escape(t) for t in tokens), key=len, reverse=True)
    return re.compile(r"(?<!\w)(%s)(?!\w)" % "|".join(escaped))

POLITICS_PHRASE_RE = build_phrase_pattern(POLITICS_PHRASES)
POLITICS_ENTITY_RE = build_token_pattern(POLITICS_ENTITIES)
POLITICS_ACTION_RE = build_token_pattern(POLITICS_ACTIONS)

SPORTS_PHRASE_RE = build_phrase_pattern(SPORTS_PHRASES)
SPORTS_ENTITY_RE = build_token_pattern(SPORTS_ENTITIES)
SPORTS_ACTION_RE = build_token_pattern(SPORTS_ACTIONS)

ENT_PHRASE_RE = build_phrase_pattern(ENT_PHRASES)
ENT_ENTITY_RE = build_token_pattern(ENT_ENTITIES)
ENT_ACTION_RE = build_token_pattern(ENT_ACTIONS)

COMPANY_PHRASE_RE = build_phrase_pattern(COMPANY_PHRASES)
COMPANY_ENTITY_RE = build_token_pattern(COMPANY_ENTITIES)
COMPANY_ACTION_RE = build_token_pattern(COMPANY_ACTIONS)


# =========================
# Helpers
# =========================

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

def extract_cashtags(text: str) -> Set[str]:
    return {m.group(0).lower() for m in CASHTAG_RE.finditer(text or "")}

def contains_phrase_compiled(text: str, pattern: Pattern[str]) -> Tuple[bool, List[str]]:
    matches = sorted(set(m.lower() for m in pattern.findall(text)))
    return (len(matches) > 0, matches)

def contains_token_compiled(text: str, pattern: Pattern[str]) -> Tuple[bool, List[str]]:
    matches = sorted(set(m.lower() for m in pattern.findall(text)))
    return (len(matches) > 0, matches)

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
        ["legacy", "screen_name"],
        ["author", "legacy", "screen_name"],
    ]
    for path in nested_candidates:
        v = deep_get(obj, path)
        if isinstance(v, str) and v.strip():
            return normalize_username(v)

    return ""

def safe_jsonl_writer(path: str):
    return open(path, "a", encoding="utf-8")

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

def build_author_prior_sets(base_dir: str = ".") -> Dict[str, Set[str]]:
    return {
        "elections_politics": load_handle_file(os.path.join(base_dir, "politics_top_242.txt")),
        "sports": load_handle_file(os.path.join(base_dir, "sports_top_276.txt")),
        "entertainment": load_handle_file(os.path.join(base_dir, "entertainment_top_255.txt")),
        "tech": load_handle_file(os.path.join(base_dir, "tech_top_195.txt")),
        "financial": load_handle_file(os.path.join(base_dir, "financial_top_288.txt")),
    }


# =========================
# Generic scoring
# =========================

def score_category(
    text_norm: str,
    raw_text: str,
    username: str,
    phrase_re: Pattern[str],
    entity_re: Pattern[str],
    action_re: Pattern[str],
    hashtags_set: Set[str],
    cashtags_set: Optional[Set[str]] = None,
    prior_strong_set: Optional[Set[str]] = None,
    prior_weak_set: Optional[Set[str]] = None,
) -> Tuple[float, Dict[str, Any]]:
    hashtags = extract_hashtags(raw_text)
    cashtags = extract_cashtags(raw_text)

    phrase_hit, matched_phrases = contains_phrase_compiled(text_norm, phrase_re)
    entity_hit, matched_entities = contains_token_compiled(text_norm, entity_re)
    action_hit, matched_actions = contains_token_compiled(text_norm, action_re)
    matched_hashtags = sorted(list(hashtags & hashtags_set))
    matched_cashtags = sorted(list(cashtags & cashtags_set)) if cashtags_set else []

    score = 0.0
    author_prior_hit = False

    if phrase_hit:
        score += SCORE_PHRASE
    if entity_hit:
        score += SCORE_ENTITY
    if action_hit:
        score += SCORE_ACTION
    if matched_hashtags:
        score += SCORE_HASHTAG
    if matched_cashtags:
        score += SCORE_CASHTAG

    if prior_strong_set and username and username in prior_strong_set:
        score += KOL_PRIOR_BOOST_STRONG
        author_prior_hit = True

    if prior_weak_set and username and username in prior_weak_set:
        score += KOL_PRIOR_BOOST_WEAK
        author_prior_hit = True

    details = {
        "score": round(score, 3),
        "matched_phrases": matched_phrases,
        "matched_entities": matched_entities,
        "matched_actions": matched_actions,
        "matched_hashtags": matched_hashtags,
        "matched_cashtags": matched_cashtags,
        "author_prior_hit": author_prior_hit,
    }
    return score, details


# =========================
# Classification
# =========================

def classify_tweet(raw_text: str, username: str, author_priors: Dict[str, Set[str]]) -> Dict[str, Dict[str, Any]]:
    text_norm = normalize_text(raw_text)
    out: Dict[str, Dict[str, Any]] = {}

    p_score, p_details = score_category(
        text_norm=text_norm,
        raw_text=raw_text,
        username=username,
        phrase_re=POLITICS_PHRASE_RE,
        entity_re=POLITICS_ENTITY_RE,
        action_re=POLITICS_ACTION_RE,
        hashtags_set=POLITICS_HASHTAGS,
        prior_strong_set=author_priors["elections_politics"],
    )
    if p_score >= MIN_CATEGORY_SCORE:
        out["elections_politics"] = p_details

    s_score, s_details = score_category(
        text_norm=text_norm,
        raw_text=raw_text,
        username=username,
        phrase_re=SPORTS_PHRASE_RE,
        entity_re=SPORTS_ENTITY_RE,
        action_re=SPORTS_ACTION_RE,
        hashtags_set=SPORTS_HASHTAGS,
        prior_strong_set=author_priors["sports"],
    )
    if s_score >= MIN_CATEGORY_SCORE:
        out["sports"] = s_details

    e_score, e_details = score_category(
        text_norm=text_norm,
        raw_text=raw_text,
        username=username,
        phrase_re=ENT_PHRASE_RE,
        entity_re=ENT_ENTITY_RE,
        action_re=ENT_ACTION_RE,
        hashtags_set=ENT_HASHTAGS,
        prior_strong_set=author_priors["entertainment"],
    )
    if e_score >= MIN_CATEGORY_SCORE:
        out["entertainment"] = e_details

    c_score, c_details = score_category(
        text_norm=text_norm,
        raw_text=raw_text,
        username=username,
        phrase_re=COMPANY_PHRASE_RE,
        entity_re=COMPANY_ENTITY_RE,
        action_re=COMPANY_ACTION_RE,
        hashtags_set=COMPANY_HASHTAGS,
        cashtags_set=COMPANY_CASHTAGS,
        prior_strong_set=author_priors["tech"],
        prior_weak_set=author_priors["financial"],
    )
    if c_score >= MIN_CATEGORY_SCORE:
        out["company_product_announcements"] = c_details

    return out


# =========================
# Debug sampling
# =========================

def sample_debug(input_path: str, prior_dir: str = ".", limit: int = 2000, show_matches: int = 30, show_nonmatches: int = 10):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    author_priors = build_author_prior_sets(prior_dir)

    total = 0
    no_text = 0
    matched_count = 0
    nonmatched_count = 0

    shown_matches = 0
    shown_nonmatches = 0

    category_hits = defaultdict(int)

    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if total >= limit:
                break

            raw_line = line.rstrip("\n")
            if not raw_line.strip():
                continue

            total += 1

            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            raw_text = get_text_from_record(obj)
            username = get_username_from_record(obj)

            if not raw_text:
                no_text += 1
                continue

            matched = classify_tweet(raw_text, username, author_priors)

            if matched:
                matched_count += 1
                for cat in matched.keys():
                    category_hits[cat] += 1

                if shown_matches < show_matches:
                    print("\n=== MATCH ===")
                    print("USER:", username)
                    print("TEXT:", raw_text[:500])
                    print("MATCHED:", json.dumps(matched, ensure_ascii=False, indent=2))
                    shown_matches += 1
            else:
                nonmatched_count += 1
                if shown_nonmatches < show_nonmatches:
                    print("\n=== NO MATCH ===")
                    print("USER:", username)
                    print("TEXT:", raw_text[:500])
                    shown_nonmatches += 1

    print("\n[DEBUG SUMMARY]")
    print("total checked     :", total)
    print("no_text           :", no_text)
    print("matched           :", matched_count)
    print("not matched       :", nonmatched_count)
    print("per-category hits :")
    for cat in CATEGORY_TO_FILE.keys():
        print(f"  - {cat}: {category_hits[cat]}")


# =========================
# Multiprocessing helpers
# =========================

_AUTHOR_PRIORS: Optional[Dict[str, Set[str]]] = None

def init_worker(author_priors: Dict[str, Set[str]]) -> None:
    global _AUTHOR_PRIORS
    _AUTHOR_PRIORS = author_priors

def read_in_chunks(file_obj, chunk_size: int):
    while True:
        chunk = list(islice(file_obj, chunk_size))
        if not chunk:
            break
        yield chunk

def process_lines_chunk(lines: List[str]) -> List[Dict[str, Any]]:
    if _AUTHOR_PRIORS is None:
        raise RuntimeError("Worker not initialized with author priors")

    results: List[Dict[str, Any]] = []

    for line in lines:
        raw_line = line.rstrip("\n")
        if not raw_line.strip():
            continue

        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            results.append({
                "type": "json_error",
                "raw_line": raw_line,
            })
            continue

        raw_text = get_text_from_record(obj)
        username = get_username_from_record(obj)

        if not raw_text:
            results.append({
                "type": "no_text",
                "raw_line": raw_line,
            })
            continue

        matched = classify_tweet(raw_text, username, _AUTHOR_PRIORS)

        if not matched:
            results.append({
                "type": "uncategorized",
                "raw_line": raw_line,
            })
            continue

        categories = sorted(matched.keys())
        obj["_matched_categories"] = categories
        obj["_category_match_details"] = matched
        obj["_author_username_normalized"] = username

        results.append({
            "type": "categorized",
            "categories": categories,
            "serialized": json.dumps(obj, ensure_ascii=False),
        })

    return results


# =========================
# Main processing
# =========================

def process_file_parallel(
    input_path: str,
    backup: bool = True,
    prior_dir: str = ".",
    num_workers: int = 4,
    chunk_size: int = 5000,
) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    author_priors = build_author_prior_sets(prior_dir)

    print("[INFO] Loaded author prior sizes:")
    for k, v in author_priors.items():
        print(f"  - {k}: {len(v)}")

    backup_path = input_path + ".bak"
    temp_uncategorized_path = input_path + ".tmp"

    if backup:
        shutil.copy2(input_path, backup_path)
        print(f"[INFO] Backup created: {backup_path}")

    writers = {
        cat: safe_jsonl_writer(filename)
        for cat, filename in CATEGORY_TO_FILE.items()
    }

    total = 0
    categorized = 0
    uncategorized = 0
    category_hits = defaultdict(int)
    multi_label_count = 0
    no_text_count = 0
    json_error_count = 0

    ctx = mp.get_context("spawn")

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(temp_uncategorized_path, "w", encoding="utf-8") as unc_out, \
         ctx.Pool(
             processes=num_workers,
             initializer=init_worker,
             initargs=(author_priors,),
         ) as pool:

        chunk_iter = read_in_chunks(fin, chunk_size)

        for chunk_results in pool.imap(process_lines_chunk, chunk_iter, chunksize=1):
            for item in chunk_results:
                item_type = item["type"]
                total += 1

                if item_type == "categorized":
                    categories = item["categories"]
                    serialized = item["serialized"]

                    if len(categories) > 1:
                        multi_label_count += 1

                    for cat in categories:
                        writers[cat].write(serialized + "\n")
                        category_hits[cat] += 1

                    categorized += 1

                elif item_type == "json_error":
                    unc_out.write(item["raw_line"] + "\n")
                    uncategorized += 1
                    json_error_count += 1

                elif item_type == "no_text":
                    unc_out.write(item["raw_line"] + "\n")
                    uncategorized += 1
                    no_text_count += 1

                elif item_type == "uncategorized":
                    unc_out.write(item["raw_line"] + "\n")
                    uncategorized += 1

            if total % 100000 == 0:
                print(
                    f"[INFO] processed={total}, categorized={categorized}, "
                    f"uncategorized={uncategorized}, multi_label={multi_label_count}"
                )

    for w in writers.values():
        w.close()

    os.replace(temp_uncategorized_path, input_path)

    print("\n[DONE]")
    print(f"Total processed      : {total}")
    print(f"Categorized removed  : {categorized}")
    print(f"Left in merged_all   : {uncategorized}")
    print(f"No text count        : {no_text_count}")
    print(f"JSON error count     : {json_error_count}")
    print(f"Multi-label count    : {multi_label_count}")
    print("Per-category written :")
    for cat in CATEGORY_TO_FILE.keys():
        print(f"  - {cat}: {category_hits[cat]}")
    print("Output files:")
    for cat, path in CATEGORY_TO_FILE.items():
        print(f"  - {cat}: {path}")


# =========================
# CLI
# =========================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="merged_all.jsonl",
        help="Path to merged_all.jsonl"
    )
    parser.add_argument(
        "--prior-dir",
        type=str,
        default=".",
        help="Directory containing prior txt files"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable backup creation"
    )
    parser.add_argument(
        "--sample-debug",
        type=int,
        default=0,
        help="Run debug mode on first N records only"
    )
    parser.add_argument(
        "--show-matches",
        type=int,
        default=30,
        help="How many matched examples to print in debug mode"
    )
    parser.add_argument(
        "--show-nonmatches",
        type=int,
        default=10,
        help="How many non-matched examples to print in debug mode"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Number of worker processes for full run"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Number of lines per chunk sent to each worker"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if args.sample_debug > 0:
        sample_debug(
            input_path=args.input,
            prior_dir=args.prior_dir,
            limit=args.sample_debug,
            show_matches=args.show_matches,
            show_nonmatches=args.show_nonmatches,
        )
        return

    process_file_parallel(
        input_path=args.input,
        backup=not args.no_backup,
        prior_dir=args.prior_dir,
        num_workers=args.workers,
        chunk_size=args.chunk_size,
    )

if __name__ == "__main__":
    main()