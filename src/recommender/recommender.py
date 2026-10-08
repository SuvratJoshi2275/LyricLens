"""Semantic + thematic music recommendation engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from collaborative import (
    DEFAULT_PROFILES_PATH,
    DEFAULT_TRAIN_INTERACTIONS_PATH,
    build_positive_lookup,
    user_profile_from_song_ids,
)
from embedding import embed_single_text
from emotion_theme import detect_emotion, detect_theme, emotion_similarity, theme_similarity
from preprocess import clean_lyrics


SEMANTIC_WEIGHT = 0.62
EMOTION_WEIGHT = 0.14
THEME_WEIGHT = 0.14
CROSS_GENRE_WEIGHT = 0.10
DEFAULT_HYBRID_MODEL_PATH = Path("artifacts/hybrid_model.pt")


@dataclass(frozen=True)
class FusionWeights:
    """Configurable score fusion weights for content and hybrid ranking."""

    collaborative: float = 0.35
    semantic: float = 0.35
    emotion: float = 0.12
    theme: float = 0.12
    cross_genre: float = 0.06

    @classmethod
    def content_default(cls) -> "FusionWeights":
        return cls(
            collaborative=0.0,
            semantic=SEMANTIC_WEIGHT,
            emotion=EMOTION_WEIGHT,
            theme=THEME_WEIGHT,
            cross_genre=CROSS_GENRE_WEIGHT,
        )

    @classmethod
    def hybrid_default(cls) -> "FusionWeights":
        return cls()

    def without_collaborative(self) -> "FusionWeights":
        return FusionWeights(
            collaborative=0.0,
            semantic=self.semantic,
            emotion=self.emotion,
            theme=self.theme,
            cross_genre=self.cross_genre,
        ).normalized()

    def normalized(self) -> "FusionWeights":
        values = [
            max(0.0, float(self.collaborative)),
            max(0.0, float(self.semantic)),
            max(0.0, float(self.emotion)),
            max(0.0, float(self.theme)),
            max(0.0, float(self.cross_genre)),
        ]
        total = sum(values)
        if total <= 0:
            return self.content_default()
        return FusionWeights(
            collaborative=values[0] / total,
            semantic=values[1] / total,
            emotion=values[2] / total,
            theme=values[3] / total,
            cross_genre=values[4] / total,
        )


@dataclass(frozen=True)
class QueryContext:
    song_name: str
    artist: str
    genre: str
    emotion: str
    theme: str
    lyrics_preview: str = ""
    source_song_ids: tuple[int, ...] = ()


class TrainedHybridScorer:
    """Thin inference wrapper around the saved PyTorch hybrid model."""

    def __init__(self, model_path: Path | str, embeddings: np.ndarray, expected_songs: int):
        self.model_path = Path(model_path)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.user_features: np.ndarray | None = None
        self.song_metadata: np.ndarray | None = None
        self.lyric_embeddings: np.ndarray | None = None
        self.model: Any | None = None
        self.metadata: dict[str, Any] = {}
        self.error: str | None = None

        if not self.model_path.exists():
            self.error = f"Hybrid model not found at {self.model_path}"
            return

        try:
            from hybrid_model import load_hybrid_checkpoint

            model, metadata = load_hybrid_checkpoint(self.model_path)
            if model.config.num_songs != expected_songs:
                raise ValueError(
                    f"model expects {model.config.num_songs} songs, but dataset has {expected_songs}"
                )
            self.user_features, self.song_metadata, self.lyric_embeddings = self._load_feature_store(model, metadata)
            self.model = model
            self.metadata = metadata
        except Exception as exc:
            self.error = str(exc)

    def _load_feature_store(self, model: Any, metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data_artifacts = metadata.get("data_artifacts", {})
        feature_store_path = data_artifacts.get("feature_store_path") if isinstance(data_artifacts, dict) else None
        if feature_store_path and Path(feature_store_path).exists():
            feature_store = np.load(feature_store_path)
            user_features = np.asarray(feature_store["user_features"], dtype=np.float32)
            song_metadata = np.asarray(feature_store["song_metadata"], dtype=np.float32)
            lyric_embeddings = np.asarray(feature_store["lyric_embeddings"], dtype=np.float32)
        else:
            user_features = np.zeros((model.config.num_users, model.config.user_feature_dim), dtype=np.float32)
            song_metadata = np.zeros((model.config.num_songs, model.config.metadata_dim), dtype=np.float32)
            lyric_embeddings = self.embeddings

        if user_features.shape != (model.config.num_users, model.config.user_feature_dim):
            raise ValueError("Saved user feature matrix does not match model config.")
        if song_metadata.shape != (model.config.num_songs, model.config.metadata_dim):
            raise ValueError("Saved song metadata matrix does not match model config.")
        if lyric_embeddings.shape != (model.config.num_songs, model.config.lyric_embedding_dim):
            raise ValueError("Saved lyric embedding matrix does not match model config.")
        return user_features, song_metadata, lyric_embeddings

    @property
    def available(self) -> bool:
        return self.model is not None

    @property
    def num_users(self) -> int:
        if self.model is None:
            return 0
        return int(self.model.config.num_users)

    def score_user(self, user_id: int) -> np.ndarray:
        if self.model is None:
            raise ValueError(self.error or "Hybrid model is not available.")
        if self.user_features is None or self.song_metadata is None or self.lyric_embeddings is None:
            raise ValueError("Hybrid feature store is not available.")
        return self.model.score_all_songs(
            int(user_id),
            user_features=self.user_features,
            song_metadata=self.song_metadata,
            lyric_embeddings=self.lyric_embeddings,
        )


class MusicRecommender:
    """Score songs by semantics, emotional state, theme, and cross-genre contrast."""

    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        model: Any | None = None,
        hybrid_model_path: Path | str | None = DEFAULT_HYBRID_MODEL_PATH,
    ):
        if len(df) != len(embeddings):
            raise ValueError("DataFrame and embeddings must have the same number of rows.")

        self.df = df.reset_index(drop=True)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.model = model
        self.hybrid_scorer = (
            TrainedHybridScorer(hybrid_model_path, self.embeddings, expected_songs=len(self.df))
            if hybrid_model_path
            else None
        )
        self.train_interactions = self._load_interactions(DEFAULT_TRAIN_INTERACTIONS_PATH)
        self.user_positive_lookup = build_positive_lookup(self.train_interactions)
        self.synthetic_profiles = self._load_profiles(DEFAULT_PROFILES_PATH)

    @staticmethod
    def _load_interactions(path: Path) -> pd.DataFrame:
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame(columns=["user_id", "song_id", "interaction", "play_count", "preference_score"])

    @staticmethod
    def _load_profiles(path: Path) -> pd.DataFrame:
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame(columns=["user_id", "preferred_emotions", "preferred_themes", "preferred_genres"])

    @property
    def hybrid_available(self) -> bool:
        return bool(self.hybrid_scorer and self.hybrid_scorer.available)

    @property
    def hybrid_status(self) -> str:
        if self.hybrid_available:
            assert self.hybrid_scorer is not None
            return f"trained model loaded ({self.hybrid_scorer.num_users} users)"
        if self.hybrid_scorer and self.hybrid_scorer.error:
            return self.hybrid_scorer.error
        return "hybrid model not configured"

    def available_user_ids(self) -> list[int]:
        if not self.hybrid_available:
            return []
        assert self.hybrid_scorer is not None
        return list(range(self.hybrid_scorer.num_users))

    def user_history_song_ids(self, user_id: int, limit: int | None = None) -> list[int]:
        """Return train-positive song IDs for a known synthetic user."""
        if self.train_interactions.empty:
            return []
        user_rows = self.train_interactions[self.train_interactions["user_id"].astype(int) == int(user_id)].copy()
        if user_rows.empty:
            return []
        sort_columns = [col for col in ["preference_score", "play_count"] if col in user_rows.columns]
        if sort_columns:
            user_rows = user_rows.sort_values(sort_columns, ascending=False)
        song_ids = user_rows["song_id"].astype(int).tolist()
        return song_ids[:limit] if limit else song_ids

    def user_profile(self, song_ids: Iterable[int]) -> dict[str, pd.DataFrame]:
        return user_profile_from_song_ids(self.df, song_ids)

    def search_songs(self, query: str, limit: int = 50) -> pd.DataFrame:
        """Search songs by title or artist."""
        query = query.strip().lower()
        if not query:
            return self.df.head(limit)

        mask = (
            self.df["song_name"].str.lower().str.contains(query, regex=False)
            | self.df["artist"].str.lower().str.contains(query, regex=False)
        )
        return self.df[mask].head(limit)

    def find_song_index(self, song_name: str) -> int:
        """Find a song by exact case-insensitive name, falling back to contains matching."""
        query = song_name.strip().lower()
        exact = self.df.index[self.df["song_name"].str.lower() == query].tolist()
        if exact:
            return exact[0]

        contains = self.df.index[self.df["song_name"].str.lower().str.contains(query, regex=False)].tolist()
        if contains:
            return contains[0]

        raise ValueError(f"Song not found: {song_name}")

    def context_from_index(self, index: int) -> QueryContext:
        row = self.df.iloc[index]
        return QueryContext(
            song_name=row["song_name"],
            artist=row["artist"],
            genre=row["genre"],
            emotion=row["emotion"],
            theme=row["theme"],
            lyrics_preview=self.lyrics_preview(row),
            source_song_ids=(int(index),),
        )

    def context_from_indices(self, indices: Iterable[int], label: str = "Selected Taste") -> QueryContext:
        valid_indices = [int(index) for index in indices if 0 <= int(index) < len(self.df)]
        if not valid_indices:
            raise ValueError("Select at least one song to build a taste profile.")

        subset = self.df.iloc[valid_indices]
        primary_genre = subset["genre"].mode().iloc[0]
        primary_emotion = subset["emotion"].mode().iloc[0]
        primary_theme = subset["theme"].mode().iloc[0]
        preview_titles = ", ".join(subset["song_name"].head(3).astype(str).tolist())
        if len(subset) > 3:
            preview_titles += f", +{len(subset) - 3} more"

        return QueryContext(
            song_name=label,
            artist=f"{len(valid_indices)} liked songs",
            genre=str(primary_genre),
            emotion=str(primary_emotion),
            theme=str(primary_theme),
            lyrics_preview=preview_titles,
            source_song_ids=tuple(valid_indices),
        )

    def recommend_by_song(
        self,
        song_name: str,
        top_n: int = 5,
        candidate_indices: Optional[Iterable[int]] = None,
        user_id: int | None = None,
        mode: str = "content",
        weights: FusionWeights | None = None,
    ) -> pd.DataFrame:
        """Recommend songs similar to a selected catalog song."""
        index = self.find_song_index(song_name)
        return self.recommend_by_index(
            index,
            top_n=top_n,
            candidate_indices=candidate_indices,
            user_id=user_id,
            mode=mode,
            weights=weights,
        )

    def recommend_by_index(
        self,
        index: int,
        top_n: int = 5,
        candidate_indices: Optional[Iterable[int]] = None,
        user_id: int | None = None,
        mode: str = "content",
        weights: FusionWeights | None = None,
    ) -> pd.DataFrame:
        """Recommend songs similar to a selected row index."""
        query = self.context_from_index(index)
        collaborative_scores, effective_weights, effective_mode = self._collaborative_context(
            user_id=user_id,
            mode=mode,
            weights=weights,
        )
        return self._rank(
            query_embedding=self.embeddings[index],
            query=query,
            top_n=top_n,
            exclude_index=index,
            candidate_indices=candidate_indices,
            collaborative_scores=collaborative_scores,
            weights=effective_weights,
            mode_label=effective_mode,
        )

    def recommend_for_user_taste(
        self,
        selected_indices: Iterable[int] | None = None,
        user_id: int | None = None,
        mode: str = "hybrid",
        top_n: int = 5,
        candidate_indices: Optional[Iterable[int]] = None,
        weights: FusionWeights | None = None,
    ) -> tuple[QueryContext, pd.DataFrame, str]:
        """Recommend from a multi-song taste profile with optional hybrid scoring."""
        selected = [int(index) for index in (selected_indices or []) if 0 <= int(index) < len(self.df)]
        if not selected and user_id is not None:
            selected = self.user_history_song_ids(int(user_id), limit=20)
        if not selected:
            raise ValueError("Select liked songs or choose a trained user with history.")

        query = self.context_from_indices(selected, label="User Taste Profile")
        query_embedding = np.mean(self.embeddings[selected], axis=0)
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        collaborative_scores, effective_weights, effective_mode = self._collaborative_context(
            user_id=user_id,
            mode=mode,
            weights=weights,
        )
        recommendations = self._rank(
            query_embedding=query_embedding,
            query=query,
            top_n=top_n,
            exclude_index=None,
            exclude_indices=selected,
            candidate_indices=candidate_indices,
            collaborative_scores=collaborative_scores,
            weights=effective_weights,
            mode_label=effective_mode,
        )
        return query, recommendations, effective_mode

    def recommend_by_custom_lyrics(
        self,
        lyrics: str,
        genre: str = "custom",
        top_n: int = 5,
        candidate_indices: Optional[Iterable[int]] = None,
    ) -> tuple[QueryContext, pd.DataFrame]:
        """Recommend songs for lyrics pasted by a user."""
        if self.model is None:
            raise ValueError("A SentenceTransformer model is required for custom lyrics recommendations.")

        cleaned = clean_lyrics(lyrics)
        if len(cleaned.split()) < 20:
            raise ValueError("Please enter at least 20 words so the semantic embedding has enough context.")

        query = QueryContext(
            song_name="Custom Lyrics",
            artist="User Input",
            genre=genre.lower().strip() or "custom",
            emotion=detect_emotion(cleaned),
            theme=detect_theme(cleaned),
            lyrics_preview=self.make_preview(cleaned),
        )
        query_embedding = embed_single_text(self.model, cleaned)
        recommendations = self._rank(
            query_embedding=query_embedding,
            query=query,
            top_n=top_n,
            exclude_index=None,
            candidate_indices=candidate_indices,
            collaborative_scores=None,
            weights=FusionWeights.content_default(),
            mode_label="content-based",
        )
        return query, recommendations

    def _collaborative_context(
        self,
        user_id: int | None,
        mode: str,
        weights: FusionWeights | None,
    ) -> tuple[np.ndarray | None, FusionWeights, str]:
        requested_hybrid = mode.lower().strip() == "hybrid"
        if requested_hybrid and user_id is not None and self.hybrid_available:
            assert self.hybrid_scorer is not None
            scores = self.hybrid_scorer.score_user(int(user_id))
            return scores, (weights or FusionWeights.hybrid_default()).normalized(), "hybrid"

        if weights is not None:
            return None, weights.without_collaborative(), "content-based"
        return None, FusionWeights.content_default(), "content-based"

    def _rank(
        self,
        query_embedding: np.ndarray,
        query: QueryContext,
        top_n: int,
        exclude_index: Optional[int],
        candidate_indices: Optional[Iterable[int]],
        collaborative_scores: np.ndarray | None = None,
        weights: FusionWeights | None = None,
        mode_label: str = "content-based",
        exclude_indices: Optional[Iterable[int]] = None,
    ) -> pd.DataFrame:
        weights = (weights or FusionWeights.content_default()).normalized()
        semantic_scores = cosine_similarity([query_embedding], self.embeddings)[0]
        ranked = self.df.copy()
        ranked["semantic_similarity"] = semantic_scores
        ranked["emotion_similarity"] = ranked["emotion"].apply(lambda value: emotion_similarity(value, query.emotion))
        ranked["theme_similarity"] = ranked["theme"].apply(lambda value: theme_similarity(value, query.theme))
        ranked["genre_diff"] = (ranked["genre"] != query.genre).astype(int)
        ranked["cross_genre_bonus"] = ranked["genre_diff"].astype(float)

        if collaborative_scores is not None:
            collaborative_scores = np.asarray(collaborative_scores, dtype=np.float32)
            if len(collaborative_scores) != len(ranked):
                raise ValueError("Collaborative score vector length must match the song catalog.")
            ranked["collaborative_score"] = collaborative_scores
        else:
            ranked["collaborative_score"] = 0.0

        ranked["final_score"] = (
            weights.collaborative * ranked["collaborative_score"]
            + weights.semantic * ranked["semantic_similarity"]
            + weights.emotion * ranked["emotion_similarity"]
            + weights.theme * ranked["theme_similarity"]
            + weights.cross_genre * ranked["cross_genre_bonus"]
        )
        ranked["recommendation_mode"] = mode_label

        if candidate_indices is not None:
            allowed = set(int(index) for index in candidate_indices)
            ranked = ranked[ranked.index.isin(allowed)].copy()

        if exclude_index is not None and exclude_index in ranked.index:
            ranked.loc[exclude_index, "final_score"] = -np.inf
        if exclude_indices is not None:
            for index in exclude_indices:
                if int(index) in ranked.index:
                    ranked.loc[int(index), "final_score"] = -np.inf

        ranked = ranked.sort_values("final_score", ascending=False).head(top_n).copy()
        ranked["song_id"] = ranked.index.astype(int)
        ranked["lyrics_preview"] = ranked.apply(self.lyrics_preview, axis=1)
        ranked["explanation"] = ranked.apply(lambda row: self._explain(query, row), axis=1)

        display_columns = [
            "song_id",
            "song_name",
            "artist",
            "genre",
            "emotion",
            "theme",
            "collaborative_score",
            "semantic_similarity",
            "emotion_similarity",
            "theme_similarity",
            "genre_diff",
            "cross_genre_bonus",
            "final_score",
            "recommendation_mode",
            "lyrics_preview",
            "explanation",
        ]
        optional_columns = ["source", "word_count", "emotion_confidence", "theme_confidence"]
        return ranked[[*display_columns, *[col for col in optional_columns if col in ranked.columns]]].reset_index(drop=True)

    @staticmethod
    def make_preview(text: str, max_chars: int = 260) -> str:
        cleaned = " ".join(str(text).split())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max_chars - 3].rstrip() + "..."

    def lyrics_preview(self, row: pd.Series, max_chars: int = 260) -> str:
        source = row.get("lyrics", row.get("clean_lyrics", ""))
        return self.make_preview(str(source), max_chars=max_chars)

    @staticmethod
    def _explain(query: QueryContext, row: pd.Series) -> str:
        """Build a dynamic explanation for a recommendation."""
        cross_genre = row["genre"] != query.genre
        genre_line = (
            f"across genres ({query.genre} -> {row['genre']})"
            if cross_genre
            else f"within {row['genre']}"
        )

        if row["theme_similarity"] == 1.0:
            theme_line = f"same theme: {row['theme']}"
        elif row["theme_similarity"] > 0:
            theme_line = f"related theme: {query.theme} -> {row['theme']}"
        else:
            theme_line = f"semantic lyric similarity despite a different theme ({row['theme']})"

        if row["emotion_similarity"] == 1.0:
            emotion_line = f"same emotion: {row['emotion']}"
        elif row["emotion_similarity"] > 0:
            emotion_line = f"related emotion: {query.emotion} -> {row['emotion']}"
        else:
            emotion_line = f"different emotional color: {row['emotion']}"

        collaborative_score = float(row.get("collaborative_score", 0.0))
        mode = str(row.get("recommendation_mode", "content-based"))
        collaborative_line = (
            f"the trained user-song model predicts strong affinity ({collaborative_score:.2f}), "
            if mode == "hybrid" and collaborative_score > 0
            else ""
        )

        return (
            f"Recommended because {collaborative_line}the lyrics carry similar meaning, {theme_line}, "
            f"{emotion_line}, {genre_line}."
        )


def filter_candidate_indices(
    df: pd.DataFrame,
    genres: Optional[list[str]] = None,
    emotions: Optional[list[str]] = None,
    themes: Optional[list[str]] = None,
) -> list[int]:
    """Return row indices matching UI filters."""
    mask = pd.Series(True, index=df.index)
    if genres:
        mask &= df["genre"].isin(genres)
    if emotions:
        mask &= df["emotion"].isin(emotions)
    if themes:
        mask &= df["theme"].isin(themes)
    return df.index[mask].tolist()


def recommendations_to_records(recommendations: pd.DataFrame) -> List[Dict[str, Any]]:
    """Small helper for notebooks, APIs, or tests."""
    return recommendations.to_dict(orient="records")
