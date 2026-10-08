"""Explainable emotion and theme tagging.

This module uses weighted lexicons, phrase matching, and related-label
similarity. It is still lightweight enough for a college prototype, but it is
more robust than simple one-keyword matching.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, Mapping

import pandas as pd

from preprocess import clean_lyrics


EMOTION_LEXICON: Dict[str, Dict[str, float]] = {
    "happy": {
        "happy": 2.0,
        "joy": 2.0,
        "smile": 1.6,
        "dance": 1.2,
        "sunshine": 1.5,
        "celebrate": 1.7,
        "laugh": 1.4,
        "bright": 1.1,
        "free": 1.0,
        "glad": 1.3,
        "alive": 1.0,
    },
    "sad": {
        "sad": 2.0,
        "tears": 2.0,
        "cry": 1.8,
        "broken": 1.6,
        "empty": 1.7,
        "hollow": 1.7,
        "alone": 1.5,
        "miss": 1.3,
        "pain": 1.5,
        "blue": 1.2,
        "grief": 2.2,
        "hurt": 1.4,
        "dark": 0.9,
    },
    "angry": {
        "angry": 2.0,
        "rage": 2.1,
        "fire": 1.2,
        "fight": 1.2,
        "hate": 2.0,
        "storm": 1.0,
        "burn": 1.2,
        "shout": 1.3,
        "revenge": 2.0,
        "war": 1.4,
        "blood": 1.0,
    },
    "calm": {
        "calm": 2.0,
        "quiet": 1.8,
        "peace": 2.0,
        "soft": 1.2,
        "slow": 1.0,
        "breathe": 1.6,
        "still": 1.4,
        "gentle": 1.5,
        "river": 0.9,
        "moon": 0.9,
        "sleep": 1.0,
    },
    "motivational": {
        "rise": 1.8,
        "dream": 1.3,
        "win": 1.8,
        "hustle": 1.7,
        "strong": 1.6,
        "climb": 1.5,
        "build": 1.2,
        "goal": 1.7,
        "believe": 1.5,
        "champion": 1.8,
        "survive": 1.2,
    },
    "romantic": {
        "love": 2.0,
        "heart": 1.5,
        "kiss": 1.8,
        "darling": 1.6,
        "forever": 1.3,
        "hold": 1.0,
        "touch": 1.2,
        "beloved": 1.7,
        "together": 1.2,
        "romance": 2.0,
        "baby": 0.8,
    },
}


THEME_LEXICON: Dict[str, Dict[str, float]] = {
    "heartbreak": {
        "heartbreak": 2.4,
        "breakup": 2.3,
        "left me": 2.0,
        "goodbye": 1.8,
        "betray": 1.8,
        "broken heart": 2.2,
        "without you": 1.8,
        "farewell": 1.4,
        "door closed": 1.4,
    },
    "loneliness": {
        "alone": 1.8,
        "lonely": 2.2,
        "empty": 1.7,
        "hollow": 1.7,
        "silence": 1.6,
        "no one": 2.0,
        "distant": 1.2,
        "isolated": 2.0,
        "shadow": 1.2,
    },
    "loss/grief": {
        "death": 2.3,
        "lost": 1.7,
        "gone": 1.5,
        "miss you": 2.0,
        "grave": 2.1,
        "funeral": 2.3,
        "ashes": 1.8,
        "memory": 1.0,
        "grief": 2.4,
        "mourning": 2.2,
    },
    "guilt/regret": {
        "sorry": 1.8,
        "fault": 2.0,
        "regret": 2.3,
        "blame": 2.0,
        "forgive": 1.7,
        "mistake": 1.8,
        "apology": 2.0,
        "shame": 1.8,
        "my fault": 2.2,
        "confess": 1.4,
    },
    "self-reflection": {
        "mirror": 1.8,
        "inside": 1.2,
        "question": 1.3,
        "truth": 1.5,
        "myself": 1.8,
        "soul": 1.2,
        "thinking": 1.2,
        "learn": 1.2,
        "change": 1.2,
        "reflection": 2.0,
    },
    "ambition": {
        "ambition": 2.2,
        "dream": 1.5,
        "city": 0.8,
        "crown": 1.3,
        "goal": 1.8,
        "money": 1.2,
        "stage": 1.0,
        "legacy": 1.7,
        "future": 1.4,
        "grind": 1.5,
    },
    "struggle": {
        "struggle": 2.2,
        "battle": 1.8,
        "survive": 1.8,
        "pressure": 1.5,
        "scar": 1.5,
        "fight": 1.4,
        "pain": 1.2,
        "hard road": 1.8,
        "heavy": 0.9,
        "broken road": 1.7,
    },
}


EMOTION_RELATEDNESS: Mapping[tuple[str, str], float] = {
    ("sad", "calm"): 0.35,
    ("sad", "romantic"): 0.30,
    ("sad", "angry"): 0.25,
    ("angry", "motivational"): 0.30,
    ("happy", "romantic"): 0.45,
    ("happy", "motivational"): 0.35,
    ("calm", "romantic"): 0.30,
}


THEME_RELATEDNESS: Mapping[tuple[str, str], float] = {
    ("heartbreak", "loneliness"): 0.55,
    ("heartbreak", "loss/grief"): 0.45,
    ("heartbreak", "guilt/regret"): 0.35,
    ("loneliness", "loss/grief"): 0.35,
    ("loneliness", "self-reflection"): 0.35,
    ("loss/grief", "guilt/regret"): 0.30,
    ("guilt/regret", "self-reflection"): 0.45,
    ("struggle", "self-reflection"): 0.35,
    ("struggle", "ambition"): 0.45,
    ("ambition", "self-reflection"): 0.25,
}


def _weighted_hits(text: str, lexicon: Mapping[str, float]) -> float:
    score = 0.0
    for keyword, weight in lexicon.items():
        cleaned_keyword = clean_lyrics(keyword)
        if not cleaned_keyword:
            continue
        if " " in cleaned_keyword:
            score += text.count(cleaned_keyword) * weight
        else:
            score += len(re.findall(rf"\b{re.escape(cleaned_keyword)}\b", text)) * weight
    return score


def score_labels(text: object, lexicons: Dict[str, Dict[str, float]]) -> dict[str, float]:
    """Return normalized-ish scores for all labels."""
    cleaned = clean_lyrics(text)
    raw_scores = {
        label: _weighted_hits(cleaned, weighted_terms)
        for label, weighted_terms in lexicons.items()
    }
    word_count = max(1, len(cleaned.split()))
    return {label: score / math.sqrt(word_count) for label, score in raw_scores.items()}


def classify_with_confidence(
    text: object,
    lexicons: Dict[str, Dict[str, float]],
    default_label: str,
) -> tuple[str, float]:
    """Return best label and a confidence score between 0 and 1."""
    scores = score_labels(text, lexicons)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0

    if best_score <= 0:
        return default_label, 0.20

    confidence = min(0.99, 0.45 + best_score + max(0.0, best_score - second_score))
    return best_label, round(float(confidence), 3)


def detect_emotion(text: object) -> str:
    """Classify one of the required emotion labels."""
    return classify_with_confidence(text, EMOTION_LEXICON, default_label="calm")[0]


def detect_theme(text: object) -> str:
    """Classify one of the required theme labels."""
    return classify_with_confidence(text, THEME_LEXICON, default_label="self-reflection")[0]


def emotion_similarity(left: str, right: str) -> float:
    """Exact or related emotional-state similarity."""
    if left == right:
        return 1.0
    return float(EMOTION_RELATEDNESS.get((left, right), EMOTION_RELATEDNESS.get((right, left), 0.0)))


def theme_similarity(left: str, right: str) -> float:
    """Exact or related thematic similarity."""
    if left == right:
        return 1.0
    return float(THEME_RELATEDNESS.get((left, right), THEME_RELATEDNESS.get((right, left), 0.0)))


def add_emotion_theme_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add labels and confidence scores to every song row."""
    labelled = df.copy()
    source_col = "clean_lyrics" if "clean_lyrics" in labelled.columns else "lyrics"

    emotion_results = labelled[source_col].apply(
        lambda text: classify_with_confidence(text, EMOTION_LEXICON, "calm")
    )
    theme_results = labelled[source_col].apply(
        lambda text: classify_with_confidence(text, THEME_LEXICON, "self-reflection")
    )

    labelled["emotion"] = emotion_results.apply(lambda item: item[0])
    labelled["emotion_confidence"] = emotion_results.apply(lambda item: item[1])
    labelled["theme"] = theme_results.apply(lambda item: item[0])
    labelled["theme_confidence"] = theme_results.apply(lambda item: item[1])
    return labelled
