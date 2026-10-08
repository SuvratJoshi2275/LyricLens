"""Data cleaning utilities for the music recommender."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["song_name", "artist", "lyrics", "genre"]


GENRE_ALIASES = {
    "hip-hop": ["hip hop", "hip-hop", "rap", "trap", "boom bap"],
    "rock": ["rock", "alt rock", "alternative rock", "classic rock", "hard rock", "punk"],
    "metal": ["metal", "heavy metal", "death metal", "black metal", "nu metal", "metalcore"],
    "pop": ["pop", "dance pop", "electropop", "synthpop"],
    "country": ["country"],
    "electronic": ["electronic", "edm", "house", "techno", "trance", "dubstep"],
    "folk": ["folk", "singer songwriter", "acoustic"],
    "r&b": ["r&b", "rnb", "soul"],
    "blues": ["blues"],
    "jazz": ["jazz"],
    "hindi": ["hindi", "bollywood"],
    "punjabi": ["punjabi"],
}


def validate_columns(df: pd.DataFrame, required_columns: Iterable[str] = REQUIRED_COLUMNS) -> None:
    """Raise a clear error if the dataset is missing required columns."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def normalize_genre(value: object) -> str:
    """Map noisy genre strings into stable display buckets."""
    if pd.isna(value):
        return "unknown"

    text = str(value).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = text.lower().strip()
    if not text or text in {"nan", "none", "[]"}:
        return "unknown"

    text = re.sub(r"[\[\]\"']", " ", text)
    text = re.sub(r"[_/]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    for canonical, aliases in GENRE_ALIASES.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", text) for alias in aliases):
            return canonical

    return text.split(",")[0].strip() or "unknown"


def clean_lyrics(text: object) -> str:
    """Lowercase lyrics, remove markup/noise/special characters, and normalize spaces."""
    if pd.isna(text):
        return ""

    text = str(text).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[[^\]]+\]|\([^\)]*(chorus|verse|bridge|intro|outro)[^\)]*\)", " ", text)
    text = re.sub(r"\b(embed|lyrics|contributors|translations?)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(\w+)(\s+\1\b){3,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text: str) -> int:
    """Count whitespace-delimited words after cleaning."""
    return len(text.split())


def preprocess_dataframe(df: pd.DataFrame, min_words: int = 20) -> pd.DataFrame:
    """Clean the input dataset and remove rows that are too short for embeddings."""
    validate_columns(df)

    cleaned = df.copy()
    cleaned["song_name"] = (
        cleaned["song_name"]
        .fillna("")
        .apply(lambda value: str(value).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore").strip())
    )
    cleaned["artist"] = (
        cleaned["artist"]
        .fillna("")
        .apply(lambda value: str(value).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore").strip())
    )
    cleaned["genre"] = cleaned["genre"].apply(normalize_genre)
    cleaned["lyrics"] = cleaned["lyrics"].fillna("").astype(str)

    cleaned = cleaned[
        (cleaned["song_name"] != "")
        & (cleaned["artist"] != "")
        & (cleaned["genre"] != "")
        & (cleaned["genre"] != "unknown")
        & (cleaned["lyrics"] != "")
    ].copy()

    cleaned["clean_lyrics"] = cleaned["lyrics"].apply(clean_lyrics)
    cleaned["word_count"] = cleaned["clean_lyrics"].apply(word_count)
    cleaned = cleaned[cleaned["word_count"] >= min_words].copy()
    cleaned["lyrics_hash"] = pd.util.hash_pandas_object(cleaned["clean_lyrics"], index=False).astype(str)

    cleaned = cleaned.drop_duplicates(subset=["song_name", "artist"], keep="first")
    cleaned = cleaned.drop_duplicates(subset=["lyrics_hash"], keep="first")
    cleaned = cleaned.reset_index(drop=True)
    return cleaned


@dataclass(frozen=True)
class TabularFeatureEncoder:
    """Simple numeric + low-cardinality categorical encoder for CSV features."""

    numeric_columns: list[str]
    categorical_columns: list[str]
    category_levels: dict[str, list[str]]
    means: dict[str, float]
    stds: dict[str, float]
    max_categories: int = 50

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TabularFeatureEncoder":
        return cls(
            numeric_columns=[str(value) for value in payload.get("numeric_columns", [])],
            categorical_columns=[str(value) for value in payload.get("categorical_columns", [])],
            category_levels={
                str(column): [str(level) for level in levels]
                for column, levels in dict(payload.get("category_levels", {})).items()
            },
            means={str(column): float(value) for column, value in dict(payload.get("means", {})).items()},
            stds={str(column): float(value) for column, value in dict(payload.get("stds", {})).items()},
            max_categories=int(payload.get("max_categories", 50)),
        )


@dataclass
class PreparedHybridData:
    """Aligned tensors and metadata used by training/evaluation."""

    interactions: pd.DataFrame
    train_interactions: pd.DataFrame
    validation_interactions: pd.DataFrame
    user_features: np.ndarray
    song_metadata: np.ndarray
    lyric_embeddings: np.ndarray
    user_id_to_index: dict[str, int]
    song_id_to_index: dict[str, int]
    index_to_user_id: list[str]
    index_to_song_id: list[str]
    user_feature_encoder: TabularFeatureEncoder
    song_feature_encoder: TabularFeatureEncoder
    song_catalog: pd.DataFrame


TEXT_METADATA_COLUMNS = {
    "song_name",
    "title",
    "track_name",
    "artist",
    "artist_name",
    "lyrics",
    "clean_lyrics",
    "album",
    "source",
}


def _ordered_unique(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered


def fit_tabular_feature_encoder(
    df: pd.DataFrame,
    id_column: str,
    exclude_columns: Iterable[str] | None = None,
    max_categories: int = 50,
) -> TabularFeatureEncoder:
    """Fit a lightweight feature encoder for numeric and compact categorical columns."""
    excluded = {id_column, *(exclude_columns or [])}
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    category_levels: dict[str, list[str]] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}

    for column in df.columns:
        if column in excluded:
            continue
        series = df[column]
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_ratio = float(numeric.notna().mean()) if len(series) else 0.0

        if numeric_ratio >= 0.95:
            numeric_columns.append(column)
            mean = float(numeric.mean()) if numeric.notna().any() else 0.0
            std = float(numeric.std(ddof=0)) if numeric.notna().any() else 1.0
            means[column] = mean
            stds[column] = std if std > 1e-8 else 1.0
            continue

        values = series.fillna("__missing__").astype(str)
        levels = sorted(values.unique().tolist())
        if 1 < len(levels) <= max_categories:
            categorical_columns.append(column)
            category_levels[column] = levels

    return TabularFeatureEncoder(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        category_levels=category_levels,
        means=means,
        stds=stds,
        max_categories=max_categories,
    )


def transform_tabular_features(df: pd.DataFrame, encoder: TabularFeatureEncoder) -> np.ndarray:
    """Transform a frame with a fitted encoder into a float32 feature matrix."""
    parts: list[np.ndarray] = []
    for column in encoder.numeric_columns:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce").fillna(encoder.means.get(column, 0.0))
        else:
            values = pd.Series(encoder.means.get(column, 0.0), index=df.index)
        standardized = (values.astype(float) - encoder.means.get(column, 0.0)) / encoder.stds.get(column, 1.0)
        parts.append(standardized.to_numpy(dtype=np.float32).reshape(-1, 1))

    for column in encoder.categorical_columns:
        levels = encoder.category_levels.get(column, [])
        if column in df.columns:
            values = df[column].fillna("__missing__").astype(str)
        else:
            values = pd.Series("__missing__", index=df.index)
        encoded = np.zeros((len(df), len(levels)), dtype=np.float32)
        level_to_index = {level: idx for idx, level in enumerate(levels)}
        for row_idx, value in enumerate(values):
            level_idx = level_to_index.get(str(value))
            if level_idx is not None:
                encoded[row_idx, level_idx] = 1.0
        parts.append(encoded)

    if not parts:
        return np.zeros((len(df), 1), dtype=np.float32)
    return np.concatenate(parts, axis=1).astype(np.float32)


def build_positive_lookup(interactions: pd.DataFrame) -> dict[int, set[int]]:
    """Return mapped user_idx -> positive song_idx set."""
    lookup: dict[int, set[int]] = {}
    if interactions.empty:
        return lookup
    for user_idx, group in interactions.groupby("user_idx"):
        lookup[int(user_idx)] = set(group["song_idx"].astype(int).tolist())
    return lookup


def split_interactions_by_user(
    interactions: pd.DataFrame,
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split positives per user while keeping at least one train item when possible."""
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []

    for user_idx, group in interactions.groupby("user_idx", sort=True):
        shuffled = group.sample(frac=1.0, random_state=seed + int(user_idx))
        if len(shuffled) <= 1:
            train_parts.append(shuffled)
            continue
        validation_count = max(1, int(round(len(shuffled) * validation_ratio)))
        validation_count = min(validation_count, len(shuffled) - 1)
        validation_parts.append(shuffled.head(validation_count))
        train_parts.append(shuffled.iloc[validation_count:])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else interactions.head(0)
    validation_df = pd.concat(validation_parts, ignore_index=True) if validation_parts else interactions.head(0)
    return train_df.reset_index(drop=True), validation_df.reset_index(drop=True)


def prepare_hybrid_data(
    interactions: pd.DataFrame,
    user_features: pd.DataFrame,
    song_characteristics: pd.DataFrame,
    lyric_embeddings: pd.DataFrame,
    min_play_count: float = 0.0,
    confidence_alpha: float = 1.0,
    validation_ratio: float = 0.2,
    seed: int = 42,
    max_categories: int = 50,
) -> PreparedHybridData:
    """Create ID mappings, implicit preferences, confidence values, and feature matrices."""
    for frame, required in [
        (interactions, ["user_id", "song_id", "play_count"]),
        (user_features, ["user_id"]),
        (song_characteristics, ["song_id"]),
        (lyric_embeddings, ["song_id"]),
    ]:
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    interactions = interactions.copy()
    # ===== IMPROVE INTERACTION SIGNAL =====
    interactions["play_count"] = interactions["play_count"].clip(lower=0)

    # log scaling
    interactions["play_count"] = np.log1p(interactions["play_count"])

    # normalize to 0–1
    if interactions["play_count"].max() > 0:
        interactions["play_count"] = interactions["play_count"] / interactions["play_count"].max()
    user_features = user_features.copy()
    song_characteristics = song_characteristics.copy()
    lyric_embeddings = lyric_embeddings.copy()
    for frame, columns in [
        (interactions, ["user_id", "song_id"]),
        (user_features, ["user_id"]),
        (song_characteristics, ["song_id"]),
        (lyric_embeddings, ["song_id"]),
    ]:
        for column in columns:
            frame[column] = frame[column].astype(str)

    interactions["play_count"] = pd.to_numeric(interactions["play_count"], errors="coerce").fillna(0.0)
    interactions = interactions[interactions["play_count"] > float(min_play_count)].copy()
    interactions = (
        interactions.groupby(["user_id", "song_id"], as_index=False)["play_count"]
        .sum()
        .sort_values(["user_id", "song_id"])
        .reset_index(drop=True)
    )

    embedding_columns = [column for column in lyric_embeddings.columns if column != "song_id"]
    if not embedding_columns:
        raise ValueError("lyric_embeddings must contain at least one embedding column.")
    lyric_embeddings[embedding_columns] = lyric_embeddings[embedding_columns].apply(pd.to_numeric, errors="coerce")
    if lyric_embeddings[embedding_columns].isna().any().any():
        raise ValueError("lyric_embeddings contains non-numeric values after parsing.")

    valid_user_ids = set(user_features["user_id"].astype(str))
    songs_with_metadata = set(song_characteristics["song_id"].astype(str))
    songs_with_lyrics = set(lyric_embeddings["song_id"].astype(str))
    valid_song_ids = songs_with_metadata & songs_with_lyrics
    interactions = interactions[
        interactions["user_id"].isin(valid_user_ids) & interactions["song_id"].isin(valid_song_ids)
    ].copy()
    if interactions.empty:
        raise ValueError("No positive interactions remain after filtering to available users/songs.")

    index_to_user_id = _ordered_unique(user_features[user_features["user_id"].isin(interactions["user_id"])]["user_id"])
    index_to_song_id = _ordered_unique(song_characteristics[song_characteristics["song_id"].isin(valid_song_ids)]["song_id"])
    user_id_to_index = {user_id: idx for idx, user_id in enumerate(index_to_user_id)}
    song_id_to_index = {song_id: idx for idx, song_id in enumerate(index_to_song_id)}

    interactions["user_idx"] = interactions["user_id"].map(user_id_to_index)
    interactions["song_idx"] = interactions["song_id"].map(song_id_to_index)
    interactions = interactions.dropna(subset=["user_idx", "song_idx"]).copy()
    interactions["user_idx"] = interactions["user_idx"].astype(int)
    interactions["song_idx"] = interactions["song_idx"].astype(int)
    interactions["preference"] = 1.0
    interactions["confidence"] = 1.0 + float(confidence_alpha) * np.log1p(interactions["play_count"].astype(float))

    aligned_users = (
        pd.DataFrame({"user_id": index_to_user_id})
        .merge(user_features, on="user_id", how="left", sort=False)
        .reset_index(drop=True)
    )
    aligned_songs = (
        pd.DataFrame({"song_id": index_to_song_id})
        .merge(song_characteristics, on="song_id", how="left", sort=False)
        .reset_index(drop=True)
    )
    aligned_lyrics = (
        pd.DataFrame({"song_id": index_to_song_id})
        .merge(lyric_embeddings, on="song_id", how="left", sort=False)
        .reset_index(drop=True)
    )
    if aligned_lyrics[embedding_columns].isna().any().any():
        raise ValueError("Some mapped songs are missing lyric embeddings.")

    user_encoder = fit_tabular_feature_encoder(
        aligned_users,
        id_column="user_id",
        max_categories=max_categories,
    )
    song_encoder = fit_tabular_feature_encoder(
        aligned_songs,
        id_column="song_id",
        exclude_columns=TEXT_METADATA_COLUMNS,
        max_categories=max_categories,
    )
    user_feature_matrix = transform_tabular_features(aligned_users, user_encoder)
    song_feature_matrix = transform_tabular_features(aligned_songs, song_encoder)
    lyric_matrix = aligned_lyrics[embedding_columns].to_numpy(dtype=np.float32)

    train_interactions, validation_interactions = split_interactions_by_user(
        interactions,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    if train_interactions.empty:
        raise ValueError("Training split is empty. Add more interactions per user or lower validation_ratio.")

    song_catalog = aligned_songs.copy()
    song_catalog.insert(0, "song_idx", np.arange(len(song_catalog), dtype=int))

    return PreparedHybridData(
        interactions=interactions.reset_index(drop=True),
        train_interactions=train_interactions,
        validation_interactions=validation_interactions,
        user_features=user_feature_matrix,
        song_metadata=song_feature_matrix,
        lyric_embeddings=lyric_matrix,
        user_id_to_index=user_id_to_index,
        song_id_to_index=song_id_to_index,
        index_to_user_id=index_to_user_id,
        index_to_song_id=index_to_song_id,
        user_feature_encoder=user_encoder,
        song_feature_encoder=song_encoder,
        song_catalog=song_catalog,
    )


def save_prepared_hybrid_data(prepared: PreparedHybridData, output_dir: Path | str) -> dict[str, str]:
    """Persist feature matrices, mappings, and train/validation splits for reuse."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_store_path = output_dir / "hybrid_feature_store.npz"
    metadata_path = output_dir / "hybrid_data_metadata.json"
    train_path = output_dir / "hybrid_train_interactions.csv"
    validation_path = output_dir / "hybrid_validation_interactions.csv"
    catalog_path = output_dir / "hybrid_song_catalog.csv"

    np.savez_compressed(
        feature_store_path,
        user_features=prepared.user_features,
        song_metadata=prepared.song_metadata,
        lyric_embeddings=prepared.lyric_embeddings,
    )
    prepared.train_interactions.to_csv(train_path, index=False)
    prepared.validation_interactions.to_csv(validation_path, index=False)
    prepared.song_catalog.to_csv(catalog_path, index=False)

    metadata = {
        "user_id_to_index": prepared.user_id_to_index,
        "song_id_to_index": prepared.song_id_to_index,
        "index_to_user_id": prepared.index_to_user_id,
        "index_to_song_id": prepared.index_to_song_id,
        "user_feature_encoder": prepared.user_feature_encoder.to_dict(),
        "song_feature_encoder": prepared.song_feature_encoder.to_dict(),
        "feature_store_path": str(feature_store_path),
        "train_interactions_path": str(train_path),
        "validation_interactions_path": str(validation_path),
        "song_catalog_path": str(catalog_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return {
        "feature_store_path": str(feature_store_path),
        "metadata_path": str(metadata_path),
        "train_interactions_path": str(train_path),
        "validation_interactions_path": str(validation_path),
        "song_catalog_path": str(catalog_path),
    }


def load_prepared_hybrid_artifacts(output_dir: Path | str) -> dict[str, object]:
    """Load persisted tensors and mapped splits produced by training."""
    output_dir = Path(output_dir)
    metadata_path = output_dir / "hybrid_data_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Hybrid metadata not found at {metadata_path}. Run training first.")
    metadata = json.loads(metadata_path.read_text())
    feature_store = np.load(metadata["feature_store_path"])
    return {
        "metadata": metadata,
        "user_features": np.asarray(feature_store["user_features"], dtype=np.float32),
        "song_metadata": np.asarray(feature_store["song_metadata"], dtype=np.float32),
        "lyric_embeddings": np.asarray(feature_store["lyric_embeddings"], dtype=np.float32),
        "train_interactions": pd.read_csv(metadata["train_interactions_path"]),
        "validation_interactions": pd.read_csv(metadata["validation_interactions_path"]),
        "song_catalog": pd.read_csv(metadata["song_catalog_path"]),
    }