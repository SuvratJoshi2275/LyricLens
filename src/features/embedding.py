"""Sentence-BERT embedding utilities with efficient cache invalidation."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PRIMARY_MODEL_NAME = "all-mpnet-base-v2"
FALLBACK_MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_NAME = PRIMARY_MODEL_NAME
DEFAULT_EMBEDDING_PATH = Path("artifacts/lyrics_embeddings.pkl")


def _model_cache_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def embedding_cache_path(model_name: str) -> Path:
    return Path("artifacts") / f"lyrics_embeddings_{_model_cache_name(model_name)}.pkl"


def load_embedding_model(
    model_name: str = PRIMARY_MODEL_NAME,
    fallback_model_name: str = FALLBACK_MODEL_NAME,
    local_files_only: bool = False,
):
    """Load mpnet when available, otherwise fall back to MiniLM."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is not installed. Run: pip install -r requirements.txt"
        ) from exc

    attempts = [model_name]
    if fallback_model_name and fallback_model_name not in attempts:
        attempts.append(fallback_model_name)

    errors: list[str] = []
    for candidate in attempts:
        try:
            model = SentenceTransformer(candidate, local_files_only=local_files_only)
            setattr(model, "_selected_model_name", candidate)
            return model
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
            if not local_files_only:
                try:
                    model = SentenceTransformer(candidate, local_files_only=True)
                    setattr(model, "_selected_model_name", candidate)
                    return model
                except Exception as local_exc:
                    errors.append(f"{candidate} local cache: {local_exc}")

    raise RuntimeError(
        "Could not load a Sentence-BERT model. Connect to the internet once to "
        "download all-mpnet-base-v2 or all-MiniLM-L6-v2.\n" + "\n".join(errors[-3:])
    )


def selected_model_name(model) -> str:
    return str(getattr(model, "_selected_model_name", PRIMARY_MODEL_NAME))


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Create a stable fingerprint for cache invalidation."""
    fields = ["song_name", "artist", "genre", "clean_lyrics", "emotion", "theme"]
    available = [field for field in fields if field in df.columns]
    payload = df[available].astype(str).agg("||".join, axis=1).str.cat(sep="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_embeddings(
    model,
    texts: Iterable[str],
    show_progress_bar: bool = False,
    batch_size: int = 32,
) -> np.ndarray:
    """Compute normalized dense embeddings for lyrics."""
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def load_or_create_embeddings(
    df: pd.DataFrame,
    model,
    cache_path: Path | str | None = None,
    force_rebuild: bool = False,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """Load cached embeddings when possible, otherwise compute and persist them."""
    model_name = selected_model_name(model)
    cache_path = Path(cache_path) if cache_path else embedding_cache_path(model_name)
    fingerprint = dataset_fingerprint(df)

    if cache_path.exists() and not force_rebuild:
        with cache_path.open("rb") as file:
            cached = pickle.load(file)
        if (
            cached.get("model_name") == model_name
            and cached.get("fingerprint") == fingerprint
            and "embeddings" in cached
        ):
            return np.asarray(cached["embeddings"], dtype=np.float32)

    embeddings = compute_embeddings(model, df["clean_lyrics"].tolist(), show_progress_bar=show_progress_bar)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as file:
        pickle.dump(
            {
                "model_name": model_name,
                "fingerprint": fingerprint,
                "embeddings": embeddings,
            },
            file,
        )
    return embeddings


def load_cached_embeddings(
    df: pd.DataFrame,
    model_name: str | None = None,
    artifacts_dir: Path | str = "artifacts",
) -> tuple[np.ndarray, dict[str, object]]:
    """Load a valid cached embedding matrix without instantiating SBERT.

    Training and evaluation only need the precomputed vectors. This helper
    keeps those scripts fast and offline-friendly when an embedding cache
    already matches the current dataset fingerprint.
    """
    fingerprint = dataset_fingerprint(df)
    artifacts_dir = Path(artifacts_dir)
    candidates = []
    if model_name:
        candidates.append(embedding_cache_path(model_name))
    candidates.extend(sorted(artifacts_dir.glob("lyrics_embeddings*.pkl")))

    seen: set[Path] = set()
    for path in candidates:
        path = Path(path)
        if path in seen or not path.exists():
            continue
        seen.add(path)
        with path.open("rb") as file:
            cached = pickle.load(file)
        if not isinstance(cached, dict):
            continue
        if cached.get("fingerprint") != fingerprint or "embeddings" not in cached:
            continue
        embeddings = np.asarray(cached["embeddings"], dtype=np.float32)
        if len(embeddings) != len(df):
            continue
        return embeddings, {
            "path": str(path),
            "model_name": cached.get("model_name", "unknown"),
            "fingerprint": fingerprint,
        }

    raise FileNotFoundError(
        "No valid cached lyric embeddings found for this dataset. "
        "Run the Streamlit app once or call load_or_create_embeddings first."
    )


def embed_single_text(model, text: str) -> np.ndarray:
    """Embed one custom text into the same normalized vector space."""
    embedding = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(embedding, dtype=np.float32)[0]
