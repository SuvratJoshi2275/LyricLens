"""Dataset loading and alignment utilities.

This file now serves two use cases:

1. The original Streamlit demo catalog loader, kept below for compatibility.
2. A research-style hybrid training CSV loader for real recommender datasets.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from emotion_theme import add_emotion_theme_labels
from preprocess import REQUIRED_COLUMNS, normalize_genre, preprocess_dataframe, validate_columns


DEFAULT_HYBRID_DATA_DIR = Path("data/example_hybrid")


@dataclass(frozen=True)
class HybridCSVPaths:
    """Paths for the trainable hybrid recommender CSV inputs."""

    interactions: Path
    user_features: Path
    song_characteristics: Path
    lyric_embeddings: Path


@dataclass(frozen=True)
class RawHybridData:
    """Raw CSV frames before ID mapping and tensor conversion."""

    interactions: pd.DataFrame
    user_features: pd.DataFrame
    song_characteristics: pd.DataFrame
    lyric_embeddings: pd.DataFrame
    paths: HybridCSVPaths


def resolve_hybrid_csv_paths(
    data_dir: Path | str | None = DEFAULT_HYBRID_DATA_DIR,
    interactions_path: Path | str | None = None,
    user_features_path: Path | str | None = None,
    song_characteristics_path: Path | str | None = None,
    lyric_embeddings_path: Path | str | None = None,
) -> HybridCSVPaths:
    """Resolve either a dataset directory or explicit CSV paths.

    Expected directory layout:

    - interactions.csv: user_id, song_id, play_count
    - user_features.csv: user_id plus numeric/categorical columns
    - song_characteristics.csv: song_id plus metadata columns
    - lyric_embeddings.csv: song_id plus embedding columns, or an embedding list column
    """
    base = Path(data_dir) if data_dir is not None else DEFAULT_HYBRID_DATA_DIR
    return HybridCSVPaths(
        interactions=Path(interactions_path) if interactions_path else base / "interactions.csv",
        user_features=Path(user_features_path) if user_features_path else base / "user_features.csv",
        song_characteristics=Path(song_characteristics_path)
        if song_characteristics_path
        else base / "song_characteristics.csv",
        lyric_embeddings=Path(lyric_embeddings_path) if lyric_embeddings_path else base / "lyric_embeddings.csv",
    )


def _require_csv_columns(df: pd.DataFrame, path: Path, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _parse_embedding_cell(value: object) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(item) for item in value]
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = [part for part in text.replace(";", ",").split(",") if part.strip()]
    if not isinstance(parsed, (list, tuple, np.ndarray)):
        raise ValueError(f"Embedding value must be a list-like value, got: {text[:80]}")
    return [float(item) for item in parsed]


def load_lyric_embeddings_csv(path: Path | str) -> pd.DataFrame:
    """Load lyric embeddings and normalize them to song_id + embedding_* columns."""
    path = Path(path)
    df = pd.read_csv(path)
    _require_csv_columns(df, path, ["song_id"])

    if "embedding" in df.columns:
        vectors = df["embedding"].apply(_parse_embedding_cell).tolist()
        if not vectors or any(len(vector) == 0 for vector in vectors):
            raise ValueError(f"{path} contains empty lyric embedding vectors.")
        dims = {len(vector) for vector in vectors}
        if len(dims) != 1:
            raise ValueError(f"{path} contains inconsistent lyric embedding dimensions: {sorted(dims)}")
        matrix = np.asarray(vectors, dtype=np.float32)
        columns = [f"embedding_{idx}" for idx in range(matrix.shape[1])]
        return pd.concat(
            [
                df[["song_id"]].astype({"song_id": str}).reset_index(drop=True),
                pd.DataFrame(matrix, columns=columns),
            ],
            axis=1,
        )

    embedding_columns = [column for column in df.columns if column != "song_id"]
    if not embedding_columns:
        raise ValueError(f"{path} must contain embedding columns or an 'embedding' list column.")
    numeric = df[embedding_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad_columns = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"{path} has non-numeric lyric embedding values in columns: {bad_columns}")

    normalized = pd.concat(
        [df[["song_id"]].astype({"song_id": str}).reset_index(drop=True), numeric.astype(np.float32)],
        axis=1,
    )
    rename = {column: f"embedding_{idx}" for idx, column in enumerate(embedding_columns)}
    return normalized.rename(columns=rename)


def load_hybrid_csvs(
    data_dir: Path | str | None = DEFAULT_HYBRID_DATA_DIR,
    interactions_path: Path | str | None = None,
    user_features_path: Path | str | None = None,
    song_characteristics_path: Path | str | None = None,
    lyric_embeddings_path: Path | str | None = None,
) -> RawHybridData:
    """Load the four CSV files required by the trainable hybrid model."""
    paths = resolve_hybrid_csv_paths(
        data_dir=data_dir,
        interactions_path=interactions_path,
        user_features_path=user_features_path,
        song_characteristics_path=song_characteristics_path,
        lyric_embeddings_path=lyric_embeddings_path,
    )
    missing = [str(path) for path in paths.__dict__.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing hybrid dataset CSV files:\n" + "\n".join(missing))

    interactions = pd.read_csv(paths.interactions)
    user_features = pd.read_csv(paths.user_features)
    song_characteristics = pd.read_csv(paths.song_characteristics)
    lyric_embeddings = load_lyric_embeddings_csv(paths.lyric_embeddings)

    if "interaction" in interactions.columns:
        _require_csv_columns(interactions, paths.interactions, ["user_id", "song_id", "interaction"])
    else:
        _require_csv_columns(interactions, paths.interactions, ["user_id", "song_id", "play_count"])
    _require_csv_columns(user_features, paths.user_features, ["user_id"])
    _require_csv_columns(song_characteristics, paths.song_characteristics, ["song_id"])

    interactions = interactions.copy()
    if "interaction" in interactions.columns:
        interactions["play_count"] = interactions["interaction"]
    user_features = user_features.copy()
    song_characteristics = song_characteristics.copy()
    for frame, id_columns in [
        (interactions, ["user_id", "song_id"]),
        (user_features, ["user_id"]),
        (song_characteristics, ["song_id"]),
        (lyric_embeddings, ["song_id"]),
    ]:
        for column in id_columns:
            frame[column] = frame[column].astype(str)

    return RawHybridData(
        interactions=interactions,
        user_features=user_features,
        song_characteristics=song_characteristics,
        lyric_embeddings=lyric_embeddings,
        paths=paths,
    )


ARTISTS_PATH = Path("/Users/suvratsmacbookair/Downloads/archive-3/artists.csv")
SPOTIFY_SONGS_PATH = Path("/Users/suvratsmacbookair/Downloads/archive-3/songs.csv")
CLEANED_TEST_LYRICS_PATH = Path("/Users/suvratsmacbookair/Downloads/archive-4/cleaned_test_lyrics.csv")
CLEANED_TRAIN_LYRICS_PATH = Path("/Users/suvratsmacbookair/Downloads/archive-4/cleaned_train_lyrics.csv")

UNIFIED_DATASET_PATH = Path("data/unified_songs.csv")
UNIFIED_METADATA_PATH = Path("data/unified_songs.meta.json")

CHUNK_SIZE = 75_000
MAX_ROWS_PER_SOURCE_GENRE = 90
FINAL_ROWS_PER_GENRE = 180


@dataclass(frozen=True)
class DatasetSource:
    name: str
    path: Path
    kind: str


DATASET_SOURCES = [
    DatasetSource("spotify_songs", SPOTIFY_SONGS_PATH, "spotify_songs"),
    DatasetSource("lyrics_test", CLEANED_TEST_LYRICS_PATH, "cleaned_lyrics"),
    DatasetSource("lyrics_train", CLEANED_TRAIN_LYRICS_PATH, "cleaned_lyrics"),
]


def _safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def _require_paths(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required dataset files are missing:\n"
            + "\n".join(missing)
            + "\nDownload/extract the datasets to these exact paths, then run again."
        )


def _source_signature() -> dict[str, object]:
    paths = [ARTISTS_PATH, *(source.path for source in DATASET_SOURCES)]
    _require_paths(paths)
    return {
        "paths": {
            str(path): {
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in paths
        },
        "chunk_size": CHUNK_SIZE,
        "max_rows_per_source_genre": MAX_ROWS_PER_SOURCE_GENRE,
        "final_rows_per_genre": FINAL_ROWS_PER_GENRE,
        "schema_version": 3,
    }


def _metadata_matches(signature: dict[str, object]) -> bool:
    if not UNIFIED_DATASET_PATH.exists() or not UNIFIED_METADATA_PATH.exists():
        return False
    try:
        existing = json.loads(UNIFIED_METADATA_PATH.read_text())
    except json.JSONDecodeError:
        return False
    return existing == signature


def _write_metadata(signature: dict[str, object]) -> None:
    UNIFIED_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    UNIFIED_METADATA_PATH.write_text(json.dumps(signature, indent=2))


def _safe_literal_list(value: object) -> list[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = _safe_text(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text]
    if isinstance(parsed, list):
        return [_safe_text(item) for item in parsed]
    return [_safe_text(parsed)]


def _first_artist(value: object) -> str:
    artists = _safe_literal_list(value)
    return artists[0] if artists else "Unknown Artist"


def _first_artist_id(value: object) -> str:
    artist_ids = _safe_literal_list(value)
    return artist_ids[0] if artist_ids else ""


def load_artist_lookup(path: Path = ARTISTS_PATH) -> pd.DataFrame:
    """Load artist metadata and normalize it for optional genre enrichment."""
    artists = pd.read_csv(
        path,
        usecols=["id", "name", "main_genre", "genres"],
        encoding_errors="ignore",
    )
    artists["artist_main_genre"] = artists["main_genre"].apply(normalize_genre)
    artists = artists.rename(columns={"id": "artist_id", "name": "artist_lookup_name"})
    return artists[["artist_id", "artist_lookup_name", "artist_main_genre", "genres"]]


def _sample_by_genre(df: pd.DataFrame, max_per_genre: int, random_state: int) -> pd.DataFrame:
    if df.empty:
        return df

    sampled = []
    for genre, group in df.groupby("genre", sort=True):
        if len(group) > max_per_genre:
            sampled.append(group.sample(max_per_genre, random_state=random_state))
        else:
            sampled.append(group)
    return pd.concat(sampled, ignore_index=True) if sampled else df.head(0)


def _load_spotify_songs(source: DatasetSource, artist_lookup: pd.DataFrame) -> pd.DataFrame:
    """Read the large Spotify song CSV in chunks and sample per genre."""
    usecols = ["id", "name", "artists", "lyrics", "genre", "artist_ids", "niche_genres"]
    chunks: list[pd.DataFrame] = []
    running_counts: dict[str, int] = {}

    for chunk_id, chunk in enumerate(
        pd.read_csv(
            source.path,
            usecols=usecols,
            chunksize=CHUNK_SIZE,
            encoding_errors="ignore",
        )
    ):
        chunk = chunk.rename(columns={"name": "song_name"})
        chunk["artist"] = chunk["artists"].apply(_first_artist)
        chunk["artist_id"] = chunk["artist_ids"].apply(_first_artist_id)
        chunk["genre"] = chunk["genre"].apply(normalize_genre)
        chunk = chunk.merge(artist_lookup, how="left", on="artist_id")
        missing_genre = chunk["genre"].isin(["", "unknown"])
        chunk.loc[missing_genre, "genre"] = chunk.loc[missing_genre, "artist_main_genre"].fillna("unknown")

        aligned = pd.DataFrame(
            {
                "song_name": chunk["song_name"],
                "artist": chunk["artist"].fillna(chunk["artist_lookup_name"]).fillna("Unknown Artist"),
                "lyrics": chunk["lyrics"],
                "genre": chunk["genre"],
                "source": source.name,
            }
        )
        aligned = preprocess_dataframe(aligned, min_words=20)

        keep_parts = []
        for genre, group in aligned.groupby("genre", sort=False):
            already = running_counts.get(genre, 0)
            remaining = MAX_ROWS_PER_SOURCE_GENRE - already
            if remaining <= 0:
                continue
            selected = group.head(remaining)
            running_counts[genre] = already + len(selected)
            keep_parts.append(selected)

        if keep_parts:
            chunks.append(pd.concat(keep_parts, ignore_index=True))

        known_genres = {"rock", "metal", "hip-hop", "pop", "country", "electronic", "folk", "r&b", "blues", "jazz"}
        if known_genres.issubset({genre for genre, count in running_counts.items() if count >= MAX_ROWS_PER_SOURCE_GENRE}):
            break

    if not chunks:
        return pd.DataFrame(columns=[*REQUIRED_COLUMNS, "source", "clean_lyrics", "word_count"])
    return pd.concat(chunks, ignore_index=True)


def _load_cleaned_lyrics(source: DatasetSource) -> pd.DataFrame:
    """Read lyric classification CSVs that only contain lyrics and genre."""
    chunks: list[pd.DataFrame] = []
    running_counts: dict[str, int] = {}

    for chunk_id, chunk in enumerate(
        pd.read_csv(
            source.path,
            usecols=["Unnamed: 0", "Lyric", "genre"],
            chunksize=CHUNK_SIZE,
            encoding_errors="ignore",
        )
    ):
        chunk = chunk.rename(columns={"Lyric": "lyrics", "Unnamed: 0": "source_row"})
        chunk["genre"] = chunk["genre"].apply(normalize_genre)
        chunk["song_name"] = chunk.apply(
            lambda row: f"{source.name.replace('_', ' ').title()} Song {int(row['source_row'])}",
            axis=1,
        )
        chunk["artist"] = source.name.replace("_", " ").title()
        chunk["source"] = source.name

        aligned = preprocess_dataframe(chunk[[*REQUIRED_COLUMNS, "source"]], min_words=20)
        keep_parts = []
        for genre, group in aligned.groupby("genre", sort=False):
            already = running_counts.get(genre, 0)
            remaining = MAX_ROWS_PER_SOURCE_GENRE - already
            if remaining <= 0:
                continue
            selected = group.head(remaining)
            running_counts[genre] = already + len(selected)
            keep_parts.append(selected)

        if keep_parts:
            chunks.append(pd.concat(keep_parts, ignore_index=True))

        known_genres = {"metal", "rock", "hip-hop", "pop", "country"}
        if known_genres.issubset({genre for genre, count in running_counts.items() if count >= MAX_ROWS_PER_SOURCE_GENRE}):
            break

    if not chunks:
        return pd.DataFrame(columns=[*REQUIRED_COLUMNS, "source", "clean_lyrics", "word_count"])
    return pd.concat(chunks, ignore_index=True)


def build_unified_dataset() -> pd.DataFrame:
    """Load all configured external datasets, align schemas, sample, and tag."""
    signature = _source_signature()
    artist_lookup = load_artist_lookup()

    frames: list[pd.DataFrame] = []
    for source in DATASET_SOURCES:
        if source.kind == "spotify_songs":
            frames.append(_load_spotify_songs(source, artist_lookup))
        elif source.kind == "cleaned_lyrics":
            frames.append(_load_cleaned_lyrics(source))
        else:
            raise ValueError(f"Unknown dataset source kind: {source.kind}")

    merged = pd.concat(frames, ignore_index=True)
    merged = preprocess_dataframe(merged, min_words=20)
    merged = _sample_by_genre(merged, FINAL_ROWS_PER_GENRE, random_state=42)
    merged = add_emotion_theme_labels(merged)

    UNIFIED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(UNIFIED_DATASET_PATH, index=False)
    _write_metadata(signature)
    return merged


def load_dataset(force_rebuild: bool = False) -> pd.DataFrame:
    """Load the cached unified dataset or rebuild it from all source CSVs."""
    try:
        signature = _source_signature()
    except FileNotFoundError:
        if not force_rebuild and UNIFIED_DATASET_PATH.exists():
            df = pd.read_csv(UNIFIED_DATASET_PATH)
            validate_columns(df)
            return df
        raise
    if not force_rebuild and _metadata_matches(signature):
        df = pd.read_csv(UNIFIED_DATASET_PATH)
        validate_columns(df)
        return df
    return build_unified_dataset()


def prepare_dataset(force_rebuild: bool = False) -> pd.DataFrame:
    """Load, clean, and ensure emotion/theme labels exist."""
    df = load_dataset(force_rebuild=force_rebuild)
    df = preprocess_dataframe(df, min_words=20)
    if "emotion" not in df.columns or "theme" not in df.columns:
        df = add_emotion_theme_labels(df)
    return df.reset_index(drop=True)


def dataset_summary(df: pd.DataFrame) -> dict[str, int]:
    """Return simple counts useful for UI and sanity checks."""
    return {
        "songs": int(len(df)),
        "genres": int(df["genre"].nunique()),
        "emotions": int(df["emotion"].nunique()),
        "themes": int(df["theme"].nunique()),
    }