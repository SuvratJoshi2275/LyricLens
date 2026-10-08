"""Synthetic collaborative-filtering data for the hybrid recommender.

The project does not ship real user listening histories, so this module
creates meaningful implicit-feedback data from the catalog's existing
emotion, theme, and genre signals. The synthetic users are intentionally
simple enough for a college demo while still giving the neural model a real
trainable objective.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from embedding import dataset_fingerprint
from emotion_theme import emotion_similarity, theme_similarity


DEFAULT_INTERACTIONS_PATH = Path("data/synthetic_user_interactions.csv")
DEFAULT_PROFILES_PATH = Path("data/synthetic_user_profiles.csv")
DEFAULT_INTERACTIONS_META_PATH = Path("data/synthetic_user_interactions.meta.json")
DEFAULT_TRAIN_INTERACTIONS_PATH = Path("data/synthetic_train_interactions.csv")
DEFAULT_TEST_INTERACTIONS_PATH = Path("data/synthetic_test_interactions.csv")


def _rng_choice(values: list[str], size: int, rng: np.random.Generator) -> list[str]:
    if not values:
        return []
    size = min(size, len(values))
    return [str(value) for value in rng.choice(values, size=size, replace=False)]


def _softmax(values: np.ndarray, temperature: float = 0.35) -> np.ndarray:
    shifted = (values - np.max(values)) / max(temperature, 1e-6)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def _profile_to_text(values: Iterable[str]) -> str:
    return "|".join(str(value) for value in values)


def _profile_from_text(value: object) -> list[str]:
    text = "" if pd.isna(value) else str(value)
    return [part for part in text.split("|") if part]


def _metadata(
    df: pd.DataFrame,
    num_users: int,
    positives_per_user: int,
    seed: int,
    uses_lyric_embeddings: bool,
) -> dict[str, object]:
    return {
        "dataset_fingerprint": dataset_fingerprint(df),
        "song_count": int(len(df)),
        "num_users": int(num_users),
        "positives_per_user": int(positives_per_user),
        "seed": int(seed),
        "uses_lyric_embeddings": bool(uses_lyric_embeddings),
        "schema_version": 2,
    }


def _metadata_matches(path: Path, expected: dict[str, object]) -> bool:
    if not path.exists():
        return False
    try:
        current = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return current == expected


def generate_synthetic_interactions(
    df: pd.DataFrame,
    lyric_embeddings: np.ndarray | None = None,
    num_users: int = 80,
    positives_per_user: int = 36,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create implicit user-song positives and the hidden user profiles.

    Each user receives preferred emotions, themes, genres, and a small
    cross-genre openness factor. Songs are sampled from high-affinity catalog
    items, producing positive examples that the trainer will pair with
    negative samples.
    """
    if df.empty:
        raise ValueError("Cannot generate synthetic interactions for an empty dataset.")

    rng = np.random.default_rng(seed)
    emotions = sorted(df["emotion"].dropna().astype(str).unique().tolist())
    themes = sorted(df["theme"].dropna().astype(str).unique().tolist())
    genres = sorted(df["genre"].dropna().astype(str).unique().tolist())

    row_emotions = df["emotion"].astype(str).to_numpy()
    row_themes = df["theme"].astype(str).to_numpy()
    row_genres = df["genre"].astype(str).to_numpy()
    emotion_conf = df.get("emotion_confidence", pd.Series(0.5, index=df.index)).fillna(0.5).to_numpy(dtype=float)
    theme_conf = df.get("theme_confidence", pd.Series(0.5, index=df.index)).fillna(0.5).to_numpy(dtype=float)
    normalized_embeddings: np.ndarray | None = None
    if lyric_embeddings is not None:
        normalized_embeddings = np.asarray(lyric_embeddings, dtype=np.float32)
        norms = np.linalg.norm(normalized_embeddings, axis=1, keepdims=True)
        normalized_embeddings = normalized_embeddings / np.maximum(norms, 1e-8)

    interactions: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []

    for user_id in range(num_users):
        preferred_emotions = _rng_choice(emotions, 2, rng)
        preferred_themes = _rng_choice(themes, 2, rng)
        preferred_genres = _rng_choice(genres, 2, rng)
        cross_genre_openness = float(rng.uniform(0.15, 0.7))

        emotion_affinity = np.zeros(len(df), dtype=np.float32)
        for preferred in preferred_emotions:
            emotion_affinity += np.array(
                [emotion_similarity(candidate, preferred) for candidate in row_emotions],
                dtype=np.float32,
            )
        emotion_affinity /= max(1, len(preferred_emotions))

        theme_affinity = np.zeros(len(df), dtype=np.float32)
        for preferred in preferred_themes:
            theme_affinity += np.array(
                [theme_similarity(candidate, preferred) for candidate in row_themes],
                dtype=np.float32,
            )
        theme_affinity /= max(1, len(preferred_themes))

        genre_match = np.isin(row_genres, preferred_genres).astype(np.float32)
        cross_genre_signal = (1.0 - genre_match) * (emotion_affinity + theme_affinity) * 0.5
        confidence_signal = 0.5 * emotion_conf + 0.5 * theme_conf

        base_affinity = (
            2.1 * emotion_affinity
            + 2.4 * theme_affinity
            + 1.0 * genre_match
            + cross_genre_openness * cross_genre_signal
            + 0.2 * confidence_signal
            + rng.normal(0.0, 0.08, size=len(df))
        )

        affinity = base_affinity
        if normalized_embeddings is not None:
            anchor_pool_size = min(len(df), max(positives_per_user * 4, 30))
            anchor_pool = np.argpartition(base_affinity, -anchor_pool_size)[-anchor_pool_size:]
            anchor_weights = _softmax(base_affinity[anchor_pool], temperature=0.45)
            anchor_count = min(3, len(anchor_pool))
            anchors = rng.choice(anchor_pool, size=anchor_count, replace=False, p=anchor_weights)
            semantic_affinity = normalized_embeddings @ normalized_embeddings[anchors].T
            semantic_affinity = np.max(semantic_affinity, axis=1)
            affinity = base_affinity + 1.6 * np.maximum(semantic_affinity, 0.0)

        candidate_count = min(len(df), max(positives_per_user * 8, positives_per_user))
        candidate_indices = np.argpartition(affinity, -candidate_count)[-candidate_count:]
        probabilities = _softmax(affinity[candidate_indices], temperature=0.55)
        selected_count = min(positives_per_user, len(candidate_indices))
        selected_song_ids = rng.choice(candidate_indices, size=selected_count, replace=False, p=probabilities)

        for song_id in selected_song_ids:
            raw_score = float(affinity[int(song_id)])
            implicit_strength = 1.0 / (1.0 + np.exp(-raw_score))
            play_count = int(1 + min(8, rng.poisson(1.0 + 4.0 * implicit_strength)))
            interactions.append(
                {
                    "user_id": int(user_id),
                    "song_id": int(song_id),
                    "interaction": 1,
                    "play_count": play_count,
                    "preference_score": round(raw_score, 4),
                }
            )

        profiles.append(
            {
                "user_id": int(user_id),
                "preferred_emotions": _profile_to_text(preferred_emotions),
                "preferred_themes": _profile_to_text(preferred_themes),
                "preferred_genres": _profile_to_text(preferred_genres),
                "cross_genre_openness": round(cross_genre_openness, 4),
            }
        )

    interactions_df = pd.DataFrame(interactions).sort_values(["user_id", "preference_score"], ascending=[True, False])
    profiles_df = pd.DataFrame(profiles)
    return interactions_df.reset_index(drop=True), profiles_df


def load_or_create_synthetic_interactions(
    df: pd.DataFrame,
    lyric_embeddings: np.ndarray | None = None,
    interactions_path: Path | str = DEFAULT_INTERACTIONS_PATH,
    profiles_path: Path | str = DEFAULT_PROFILES_PATH,
    meta_path: Path | str = DEFAULT_INTERACTIONS_META_PATH,
    num_users: int = 80,
    positives_per_user: int = 36,
    seed: int = 42,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached synthetic interactions or regenerate them when stale."""
    interactions_path = Path(interactions_path)
    profiles_path = Path(profiles_path)
    meta_path = Path(meta_path)
    expected = _metadata(
        df,
        num_users=num_users,
        positives_per_user=positives_per_user,
        seed=seed,
        uses_lyric_embeddings=lyric_embeddings is not None,
    )

    if (
        not force_rebuild
        and interactions_path.exists()
        and profiles_path.exists()
        and _metadata_matches(meta_path, expected)
    ):
        return pd.read_csv(interactions_path), pd.read_csv(profiles_path)

    interactions, profiles = generate_synthetic_interactions(
        df,
        lyric_embeddings=lyric_embeddings,
        num_users=num_users,
        positives_per_user=positives_per_user,
        seed=seed,
    )
    interactions_path.parent.mkdir(parents=True, exist_ok=True)
    profiles_path.parent.mkdir(parents=True, exist_ok=True)
    interactions.to_csv(interactions_path, index=False)
    profiles.to_csv(profiles_path, index=False)
    meta_path.write_text(json.dumps(expected, indent=2))
    return interactions, profiles


def split_interactions_by_user(
    interactions: pd.DataFrame,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out at least one positive per user for ranking evaluation."""
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for user_id, group in interactions.groupby("user_id", sort=True):
        shuffled = group.sample(frac=1.0, random_state=seed + int(user_id))
        if len(shuffled) <= 1:
            train_parts.append(shuffled)
            continue
        test_count = max(1, int(round(len(shuffled) * test_ratio)))
        test_parts.append(shuffled.head(test_count))
        train_parts.append(shuffled.iloc[test_count:])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else interactions.head(0)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else interactions.head(0)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_interaction_splits(
    train_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
    train_path: Path | str = DEFAULT_TRAIN_INTERACTIONS_PATH,
    test_path: Path | str = DEFAULT_TEST_INTERACTIONS_PATH,
) -> None:
    train_path = Path(train_path)
    test_path = Path(test_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    train_interactions.to_csv(train_path, index=False)
    test_interactions.to_csv(test_path, index=False)


def load_interaction_splits(
    train_path: Path | str = DEFAULT_TRAIN_INTERACTIONS_PATH,
    test_path: Path | str = DEFAULT_TEST_INTERACTIONS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = Path(train_path)
    test_path = Path(test_path)
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Train/test interaction splits are missing. Run: python trainer.py")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def build_positive_lookup(interactions: pd.DataFrame) -> dict[int, set[int]]:
    """Return user_id -> set(song_id) for masking or negative sampling."""
    lookup: dict[int, set[int]] = {}
    if interactions.empty:
        return lookup
    for user_id, group in interactions.groupby("user_id"):
        lookup[int(user_id)] = set(group["song_id"].astype(int).tolist())
    return lookup


def profile_table_from_song_ids(df: pd.DataFrame, song_ids: Iterable[int], column: str) -> pd.DataFrame:
    """Return a normalized label profile for selected songs."""
    ids = [int(song_id) for song_id in song_ids if 0 <= int(song_id) < len(df)]
    if not ids or column not in df.columns:
        return pd.DataFrame(columns=["label", "weight"])
    counts = df.iloc[ids][column].astype(str).value_counts(normalize=True)
    return counts.rename_axis("label").reset_index(name="weight")


def user_profile_from_song_ids(df: pd.DataFrame, song_ids: Iterable[int]) -> dict[str, pd.DataFrame]:
    """Build display-ready emotion/theme/genre profiles from liked songs."""
    ids = list(song_ids)
    return {
        "emotion": profile_table_from_song_ids(df, ids, "emotion"),
        "theme": profile_table_from_song_ids(df, ids, "theme"),
        "genre": profile_table_from_song_ids(df, ids, "genre"),
    }


def preferred_labels_from_profiles(profiles: pd.DataFrame, user_id: int) -> dict[str, list[str]]:
    """Decode the hidden synthetic preference labels for UI/debug views."""
    if profiles.empty or "user_id" not in profiles.columns:
        return {"emotions": [], "themes": [], "genres": []}
    matches = profiles[profiles["user_id"].astype(int) == int(user_id)]
    if matches.empty:
        return {"emotions": [], "themes": [], "genres": []}
    row = matches.iloc[0]
    return {
        "emotions": _profile_from_text(row.get("preferred_emotions", "")),
        "themes": _profile_from_text(row.get("preferred_themes", "")),
        "genres": _profile_from_text(row.get("preferred_genres", "")),
    }
