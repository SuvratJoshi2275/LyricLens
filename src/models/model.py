"""Compatibility wrapper for the hybrid recommender model."""

from hybrid_model import HybridModelConfig, HybridMusicModel, load_hybrid_checkpoint, save_hybrid_checkpoint

__all__ = [
    "HybridModelConfig",
    "HybridMusicModel",
    "load_hybrid_checkpoint",
    "save_hybrid_checkpoint",
]
