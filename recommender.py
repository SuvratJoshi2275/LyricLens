from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class LyricRecommender:
    """Original cosine-similarity recommender over the CSV's own emb_ columns.
    Unchanged — kept as the fallback when the trained hybrid model / its
    training-time feature artifacts aren't available.
    """

    backend = "cosine"

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

        self.emb_cols = [c for c in df.columns if c.startswith("emb_")]
        self.embeddings = df[self.emb_cols].values.astype("float32")

        # normalize
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.maximum(norms, 1e-8)

    def recommend(self, index, top_n=10):
        query = self.embeddings[index].reshape(1, -1)
        sims = cosine_similarity(query, self.embeddings)[0]

        idx = np.argsort(-sims)
        idx = idx[idx != index][:top_n]

        result = self.df.iloc[idx].copy()
        result["similarity"] = sims[idx]

        return result[["song_name", "artist", "similarity"]]

    def search(self, query):
        q = query.lower()
        mask = (
            self.df["song_name"].str.lower().str.contains(q, na=False) |
            self.df["artist"].str.lower().str.contains(q, na=False)
        )
        return self.df[mask].head(50)


class HybridUnavailable(Exception):
    """Raised when the trained hybrid model or its training-time feature
    artifacts can't be loaded, so the caller can fall back to LyricRecommender
    instead of faking a hybrid result."""


def _ensure_backend_modules_importable():
    """hybrid_model.py / preprocess.py / data_loader.py may live flat at the
    project root or nested under src/<subpackage>/ (e.g. src/models/hybrid_model.py,
    src/training/trainer.py) depending on how the project was laid out. Rather
    than duplicating any file, add every plausible existing directory to
    sys.path so the flat `from hybrid_model import ...` / `from preprocess
    import ...` style imports used throughout this codebase resolve no matter
    which layout is on disk.
    """
    import sys

    project_dir = Path(__file__).resolve().parent
    candidates = [project_dir, project_dir / "src"]
    src_dir = project_dir / "src"
    if src_dir.is_dir():
        candidates.extend(p for p in src_dir.iterdir() if p.is_dir())
    for c in candidates:
        c_str = str(c)
        if c.is_dir() and c_str not in sys.path:
            sys.path.insert(0, c_str)


class HybridRecommender:
    """Song-to-song recommender built on top of the trained hybrid_model.pt.

    The hybrid model itself only scores user<->song affinity (score_all_songs,
    score_cold_start) — it has no song-to-song mode. This class derives item
    similarity the legitimate way: run every song through the model's own
    encode_song() (song-id embedding + metadata encoder + lyric encoder,
    fused by the model's learned softmax gate) to get each song's fused
    representation, then rank by cosine similarity of those *learned* vectors
    instead of raw lyric embeddings. Nothing here is a fake/placeholder score.

    Requires, inside model_dir (artifacts/training/):
      - hybrid_model.pt              (trained checkpoint)
      - hybrid_data_metadata.json    (saved by save_prepared_hybrid_data)
      - hybrid_feature_store.npz     (saved by save_prepared_hybrid_data)
    These come from the *same* training run — id ordering must match, and
    this class verifies that before trusting them.
    """

    backend = "hybrid"

    def __init__(self, df, model_dir):
        model_dir = Path(model_dir)
        model_path = model_dir / "hybrid_model.pt"

        if not model_path.exists():
            raise HybridUnavailable(f"hybrid_model.pt not found at {model_path}")

        try:
            import torch
        except ImportError as e:
            raise HybridUnavailable(f"torch not installed: {e}")

        _ensure_backend_modules_importable()
        try:
            from hybrid_model import load_hybrid_checkpoint
            from preprocess import load_prepared_hybrid_artifacts
        except ImportError as e:
            raise HybridUnavailable(f"could not import model/preprocess code: {e}")

        try:
            model, checkpoint = load_hybrid_checkpoint(model_path, map_location="cpu")
        except Exception as e:
            raise HybridUnavailable(f"failed to load checkpoint: {e}")

        try:
            artifacts = load_prepared_hybrid_artifacts(model_dir)
        except Exception as e:
            raise HybridUnavailable(
                f"hybrid_data_metadata.json / hybrid_feature_store.npz not found "
                f"in {model_dir} (needed to reproduce training-time features): {e}"
            )

        ckpt_song_ids = checkpoint["index_to_song_id"]
        meta_song_ids = artifacts["metadata"]["index_to_song_id"]
        if list(ckpt_song_ids) != list(meta_song_ids):
            raise HybridUnavailable(
                "checkpoint's index_to_song_id does not match "
                "hybrid_data_metadata.json's index_to_song_id — artifacts are "
                "from different runs, refusing to mix them."
            )

        song_metadata = artifacts["song_metadata"]
        lyric_embeddings = artifacts["lyric_embeddings"]
        num_songs = len(ckpt_song_ids)
        if song_metadata.shape[0] != num_songs or lyric_embeddings.shape[0] != num_songs:
            raise HybridUnavailable("feature store row count does not match checkpoint's num_songs")

        # Fuse every song through the model's own encoder (batched, no grad).
        model.eval()
        song_vectors = np.zeros((num_songs, model.config.latent_dim), dtype=np.float32)
        batch_size = 4096
        with torch.no_grad():
            for start in range(0, num_songs, batch_size):
                end = min(start + batch_size, num_songs)
                song_ids_t = torch.arange(start, end, dtype=torch.long)
                metadata_t = torch.from_numpy(song_metadata[start:end]).float()
                lyric_t = torch.from_numpy(lyric_embeddings[start:end]).float()
                fused, _ = model.encode_song(song_ids_t, metadata_t, lyric_t)
                song_vectors[start:end] = fused.numpy()

        norms = np.linalg.norm(song_vectors, axis=1, keepdims=True)
        self.song_vectors = song_vectors / np.maximum(norms, 1e-8)

        # Map the UI's df row order <-> the checkpoint's song index order,
        # via the song_id both datasets share (verified 30,000/30,000 match).
        self.df = df.reset_index(drop=True)
        if "song_id" not in self.df.columns:
            raise HybridUnavailable("final_songs_FIXED.csv has no song_id column to align on")

        song_id_to_ckpt_idx = checkpoint["song_id_to_index"]
        df_song_ids = self.df["song_id"].astype(str)
        ckpt_idx_for_df_row = df_song_ids.map(song_id_to_ckpt_idx)

        if ckpt_idx_for_df_row.isna().any():
            missing = int(ckpt_idx_for_df_row.isna().sum())
            raise HybridUnavailable(
                f"{missing} songs in final_songs_FIXED.csv have no matching "
                f"song_id in the trained checkpoint"
            )

        self.df_pos_to_ckpt_idx = ckpt_idx_for_df_row.astype(int).to_numpy()
        # Inverse mapping: checkpoint index -> df row position (bijective here
        # since the same 30,000 songs are on both sides).
        self.ckpt_idx_to_df_pos = np.full(num_songs, -1, dtype=np.int64)
        self.ckpt_idx_to_df_pos[self.df_pos_to_ckpt_idx] = np.arange(len(self.df))

        self.model = model
        self.checkpoint = checkpoint

    def recommend(self, index, top_n=10):
        """Same signature/return shape as LyricRecommender.recommend so the
        UI's Discover flow doesn't need to change: index is a row position
        into self.df (as passed by app.py), returns song_name/artist/similarity.
        """
        ckpt_idx = int(self.df_pos_to_ckpt_idx[index])
        query = self.song_vectors[ckpt_idx].reshape(1, -1)
        sims = cosine_similarity(query, self.song_vectors)[0]

        order = np.argsort(-sims)
        order = order[order != ckpt_idx][:top_n]

        df_positions = self.ckpt_idx_to_df_pos[order]
        valid = df_positions >= 0
        df_positions = df_positions[valid]
        sims_ordered = sims[order][valid]

        result = self.df.iloc[df_positions].copy()
        result["similarity"] = sims_ordered
        return result[["song_name", "artist", "similarity"]]

    def search(self, query):
        q = query.lower()
        mask = (
            self.df["song_name"].str.lower().str.contains(q, na=False) |
            self.df["artist"].str.lower().str.contains(q, na=False)
        )
        return self.df[mask].head(50)
