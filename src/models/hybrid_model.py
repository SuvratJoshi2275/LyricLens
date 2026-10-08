"""Blueprint-aligned trainable hybrid music recommendation model.

The model keeps the architecture deliberately compact:

- user embedding + encoded user features
- song ID embedding + metadata encoder + lyric embedding encoder
- softmax gated fusion over the three song channels
- dot-product or nonlinear MLP interaction
- global, user, and song bias terms
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class HybridModelConfig:
    num_users: int
    num_songs: int
    user_feature_dim: int
    metadata_dim: int
    lyric_embedding_dim: int
    latent_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.10
    interaction_type: str = "mlp"


class MLPEncoder(nn.Module):
    """Small LayerNorm/GELU MLP used for user features, metadata, and lyrics."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float())


class HybridMusicModel(nn.Module):
    """Hybrid recommender matching the requested research blueprint."""

    def __init__(self, config: HybridModelConfig):
        super().__init__()
        if config.interaction_type not in {"dot", "mlp"}:
            raise ValueError("interaction_type must be 'dot' or 'mlp'")
        self.config = config

        self.user_embedding = nn.Embedding(config.num_users, config.latent_dim)
        self.song_embedding = nn.Embedding(config.num_songs, config.latent_dim)
        self.user_bias = nn.Embedding(config.num_users, 1)
        self.song_bias = nn.Embedding(config.num_songs, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        self.user_feature_encoder = MLPEncoder(
            input_dim=config.user_feature_dim,
            output_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.metadata_encoder = MLPEncoder(
            input_dim=config.metadata_dim,
            output_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.lyric_encoder = MLPEncoder(
            input_dim=config.lyric_embedding_dim,
            output_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )

        self.user_norm = nn.LayerNorm(config.latent_dim)
        self.song_norm = nn.LayerNorm(config.latent_dim)
        self.gate = nn.Sequential(
            nn.Linear(config.latent_dim * 3, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 3),
        )

        self.interaction_mlp = nn.Sequential(
            nn.Linear(config.latent_dim * 3, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.song_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.song_bias.weight)

    def encode_user(
        self,
        user_ids: torch.Tensor | None,
        user_features: torch.Tensor,
    ) -> torch.Tensor:
        if user_ids is None:
            user_id_vector = torch.zeros(
                (*user_features.shape[:-1], self.config.latent_dim),
                dtype=user_features.dtype,
                device=user_features.device,
            )
        else:
            user_id_vector = self.user_embedding(user_ids.long())
        encoded_features = self.user_feature_encoder(user_features)
        return self.user_norm(user_id_vector + encoded_features)

    def encode_song(
        self,
        song_ids: torch.Tensor | None,
        song_metadata: torch.Tensor,
        lyric_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if song_ids is None:
            song_id_vector = torch.zeros(
                (*song_metadata.shape[:-1], self.config.latent_dim),
                dtype=song_metadata.dtype,
                device=song_metadata.device,
            )
        else:
            song_id_vector = self.song_embedding(song_ids.long())
        metadata_vector = self.metadata_encoder(song_metadata)
        lyric_vector = self.lyric_encoder(lyric_embeddings)
        gate_logits = self.gate(torch.cat([song_id_vector, metadata_vector, lyric_vector], dim=-1))
        gate_weights = torch.softmax(gate_logits, dim=-1)

        stacked = torch.stack([song_id_vector, metadata_vector, lyric_vector], dim=1)
        song_vector = torch.sum(gate_weights.unsqueeze(-1) * stacked, dim=1)
        return self.song_norm(song_vector), gate_weights

    def interact(self, user_vector: torch.Tensor, song_vector: torch.Tensor) -> torch.Tensor:
        if self.config.interaction_type == "dot":
            return torch.sum(user_vector * song_vector, dim=-1)
        features = torch.cat(
            [
                user_vector,
                song_vector,
                user_vector * song_vector,
            ],
            dim=-1,
        )
        return self.interaction_mlp(features).squeeze(-1)

    def score_vectors(
        self,
        user_vector: torch.Tensor,
        song_vector: torch.Tensor,
        user_ids: torch.Tensor | None = None,
        song_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        interaction = self.interact(user_vector, song_vector)
        bias = self.global_bias
        if user_ids is not None:
            bias = bias + self.user_bias(user_ids.long()).squeeze(-1)
        if song_ids is not None:
            bias = bias + self.song_bias(song_ids.long()).squeeze(-1)
        return bias + interaction

    def forward(
        self,
        user_ids: torch.Tensor,
        song_ids: torch.Tensor,
        user_features: torch.Tensor,
        song_metadata: torch.Tensor,
        lyric_embeddings: torch.Tensor,
        return_gates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        user_vector = self.encode_user(user_ids, user_features)
        song_vector, gate_weights = self.encode_song(song_ids, song_metadata, lyric_embeddings)
        score = self.score_vectors(user_vector, song_vector, user_ids=user_ids, song_ids=song_ids)
        if return_gates:
            return score, gate_weights
        return score

    def score_cold_start(
        self,
        user_features: np.ndarray,
        song_metadata: np.ndarray,
        lyric_embeddings: np.ndarray,
        song_ids: np.ndarray | None = None,
        batch_size: int = 1024,
        device: str | torch.device = "cpu",
    ) -> np.ndarray:
        """Score songs for a new user from features only.

        If song_ids is omitted, the song ID embedding and song bias are also
        omitted, which supports fully new songs with metadata and lyric vectors.
        """
        self.eval()
        device = torch.device(device)
        self.to(device)
        user_features = np.asarray(user_features, dtype=np.float32).reshape(1, -1)
        song_metadata = np.asarray(song_metadata, dtype=np.float32)
        lyric_embeddings = np.asarray(lyric_embeddings, dtype=np.float32)
        song_ids_array = None if song_ids is None else np.asarray(song_ids, dtype=np.int64)
        scores: list[np.ndarray] = []

        with torch.no_grad():
            user_tensor = torch.from_numpy(user_features).to(device).float()
            user_vector = self.encode_user(None, user_tensor)
            for start in range(0, len(song_metadata), batch_size):
                end = min(start + batch_size, len(song_metadata))
                metadata_batch = torch.from_numpy(song_metadata[start:end]).to(device).float()
                lyric_batch = torch.from_numpy(lyric_embeddings[start:end]).to(device).float()
                song_id_batch = (
                    None
                    if song_ids_array is None
                    else torch.from_numpy(song_ids_array[start:end]).to(device).long()
                )
                song_vector, _ = self.encode_song(song_id_batch, metadata_batch, lyric_batch)
                user_batch = user_vector.expand(end - start, -1)
                logits = self.score_vectors(user_batch, song_vector, user_ids=None, song_ids=song_id_batch)
                scores.append(torch.sigmoid(logits).detach().cpu().numpy())
        return np.concatenate(scores).astype(np.float32)

    def score_all_songs(
        self,
        user_id: int,
        user_features: np.ndarray,
        song_metadata: np.ndarray,
        lyric_embeddings: np.ndarray,
        batch_size: int = 1024,
        device: str | torch.device = "cpu",
    ) -> np.ndarray:
        """Return sigmoid affinity scores for every song for one mapped user index."""
        if user_id < 0 or user_id >= self.config.num_users:
            raise ValueError(f"user_id {user_id} is outside trained range 0..{self.config.num_users - 1}")

        self.eval()
        device = torch.device(device)
        self.to(device)
        user_features = np.asarray(user_features, dtype=np.float32)
        song_metadata = np.asarray(song_metadata, dtype=np.float32)
        lyric_embeddings = np.asarray(lyric_embeddings, dtype=np.float32)
        if song_metadata.shape[0] != self.config.num_songs or lyric_embeddings.shape[0] != self.config.num_songs:
            raise ValueError("Feature matrices must contain one row per trained song.")

        single_user_features = torch.from_numpy(user_features[int(user_id)]).to(device).float().unsqueeze(0)
        scores: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, self.config.num_songs, batch_size):
                end = min(start + batch_size, self.config.num_songs)
                song_ids = torch.arange(start, end, dtype=torch.long, device=device)
                user_ids = torch.full_like(song_ids, int(user_id))
                user_batch = single_user_features.expand(end - start, -1)
                metadata_batch = torch.from_numpy(song_metadata[start:end]).to(device).float()
                lyric_batch = torch.from_numpy(lyric_embeddings[start:end]).to(device).float()
                logits = self.forward(user_ids, song_ids, user_batch, metadata_batch, lyric_batch)
                scores.append(torch.sigmoid(logits).detach().cpu().numpy())
        return np.concatenate(scores).astype(np.float32)


def model_config_to_dict(config: HybridModelConfig) -> dict[str, Any]:
    return asdict(config)


def model_config_from_dict(payload: dict[str, Any]) -> HybridModelConfig:
    """Load current configs and provide clear defaults for older metadata."""
    return HybridModelConfig(
        num_users=int(payload["num_users"]),
        num_songs=int(payload["num_songs"]),
        user_feature_dim=int(payload.get("user_feature_dim", 1)),
        metadata_dim=int(payload.get("metadata_dim", 1)),
        lyric_embedding_dim=int(payload["lyric_embedding_dim"]),
        latent_dim=int(payload.get("latent_dim", 128)),
        hidden_dim=int(payload.get("hidden_dim", 256)),
        dropout=float(payload.get("dropout", 0.10)),
        interaction_type=str(payload.get("interaction_type", "mlp")),
    )


def save_hybrid_checkpoint(
    model: HybridMusicModel,
    path: Path | str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist model weights plus enough metadata for inference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "model_config": model_config_to_dict(model.config),
        "state_dict": model.state_dict(),
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def load_hybrid_checkpoint(
    path: Path | str,
    map_location: str | torch.device = "cpu",
) -> tuple[HybridMusicModel, dict[str, Any]]:
    """Load a trained model checkpoint for ranking/inference."""
    checkpoint = torch.load(Path(path), map_location=map_location)
    config = model_config_from_dict(checkpoint["model_config"])
    model = HybridMusicModel(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
