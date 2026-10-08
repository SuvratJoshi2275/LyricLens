"""Training pipeline for the blueprint-aligned hybrid recommender.

Run the minimal example:

    python train.py --epochs 15

Run on your own CSV directory:

    python train.py --data-dir /path/to/dataset --epochs 20
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data_loader import DEFAULT_HYBRID_DATA_DIR, load_hybrid_csvs
from hybrid_model import HybridModelConfig, HybridMusicModel, save_hybrid_checkpoint
from preprocess import PreparedHybridData, build_positive_lookup, prepare_hybrid_data, save_prepared_hybrid_data


DEFAULT_OUTPUT_DIR = Path("artifacts")
DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "hybrid_model.pt"
DEFAULT_METRICS_PATH = DEFAULT_OUTPUT_DIR / "hybrid_training_metrics.json"


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: str = str(DEFAULT_HYBRID_DATA_DIR)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    epochs: int = 15                       # INCREASED from 8
    batch_size: int = 512                  #  INCREASED from 256
    latent_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.10
    learning_rate: float = 5e-4            #  REDUCED from 1e-3 (more stable)
    weight_decay: float = 1e-5
    negatives_per_positive: int = 20       # INCREASED from 4
    hard_negative_ratio: float = 0.40      #  NEW: 40% hard negatives
    lambda_point: float = 0.5             # REDUCED: BPR is primary signal
    lambda_rank: float = 2.0              #  INCREASED: BPR dominates
    l2_reg: float = 1e-6
    gradient_clip_norm: float = 5.0
    confidence_alpha: float = 1.0
    validation_ratio: float = 0.2
    min_play_count: float = 0.0
    max_categories: int = 50
    interaction_type: str = "mlp"
    ranking_eval_k: int = 10
    #  CRITICAL: sampled eval during training (fast + realistic)
    ranking_eval_candidates: int = 100
    max_ranking_users: int = 50           # Cap users during train-time eval
    seed: int = 42


class PairwiseInteractionDataset(Dataset):
    """Each item: one positive + N negatives (mix of hard + easy).

    WHAT CHANGED vs original:
    - Hard negatives: songs from same genre as positive, scored by
      lyric embedding cosine similarity. Top candidates sampled
      proportionally. This forces the model to learn fine-grained
      discrimination, not just genre separation.
    - hard_negative_ratio controls the split (default 0.40).
    """

    def __init__(
        self,
        positive_interactions: pd.DataFrame,
        num_songs: int,
        positives_to_avoid: dict[int, set[int]],
        negatives_per_positive: int = 20,
        hard_negative_ratio: float = 0.40,
        lyric_embeddings: np.ndarray | None = None,
        song_metadata: pd.DataFrame | None = None,  # for genre lookup
        song_genres: list[str] | None = None,        # parallel to song indices
        seed: int = 42,
    ):
        if positive_interactions.empty:
            raise ValueError("Cannot create a pairwise dataset from empty interactions.")

        rng = np.random.default_rng(seed)
        rows: list[tuple[int, int, int, float]] = []

        n_hard = int(negatives_per_positive * hard_negative_ratio)
        n_easy = negatives_per_positive - n_hard

        # Pre-build genre index for hard negatives
        genre_index: dict[str, list[int]] = {}
        if song_genres is not None:
            for idx, g in enumerate(song_genres):
                genre_index.setdefault(str(g), []).append(idx)

        # Normalize lyric embeddings once
        norm_embeddings: np.ndarray | None = None
        if lyric_embeddings is not None:
            norm_embeddings = np.asarray(lyric_embeddings, dtype=np.float32)
            norms = np.linalg.norm(norm_embeddings, axis=1, keepdims=True)
            norm_embeddings = norm_embeddings / np.maximum(norms, 1e-8)

        for row in positive_interactions.itertuples(index=False):
            user_idx = int(row.user_idx)
            pos_song_idx = int(row.song_idx)
            confidence = float(row.confidence)
            avoid = positives_to_avoid.get(user_idx, set())

            # ── HARD NEGATIVES ─────────────────────────────
            hard_negs: list[int] = []
            if n_hard > 0 and song_genres is not None and norm_embeddings is not None:
                pos_genre = song_genres[pos_song_idx] if pos_song_idx < len(song_genres) else "unknown"
                same_genre = [i for i in genre_index.get(pos_genre, []) if i not in avoid and i != pos_song_idx]
                if len(same_genre) >= n_hard:
                    # Score by lyric similarity
                    sim = norm_embeddings[same_genre] @ norm_embeddings[pos_song_idx]
                    # Take top 5*n_hard candidates then sample to avoid pure memorization
                    top_k = min(n_hard * 5, len(same_genre))
                    top_idx = np.argpartition(-sim, top_k - 1)[:top_k]
                    candidates = [same_genre[i] for i in top_idx]
                    hard_negs = list(rng.choice(candidates, size=min(n_hard, len(candidates)), replace=False))

            # ── EASY NEGATIVES ──────────────────────────────
            easy_avoid = avoid | set(hard_negs) | {pos_song_idx}
            easy_pool = [i for i in range(num_songs) if i not in easy_avoid]
            easy_negs: list[int] = []
            if easy_pool:
                easy_negs = list(rng.choice(easy_pool, size=min(n_easy, len(easy_pool)), replace=False))

            # Fallback: if hard failed, fill with easy
            all_negs = hard_negs + easy_negs
            if len(all_negs) < negatives_per_positive:
                fallback_pool = [i for i in range(num_songs) if i not in avoid and i not in set(all_negs) and i != pos_song_idx]
                extra = min(negatives_per_positive - len(all_negs), len(fallback_pool))
                if extra > 0:
                    all_negs += list(rng.choice(fallback_pool, size=extra, replace=False))

            for neg_idx in all_negs:
                rows.append((user_idx, pos_song_idx, int(neg_idx), confidence))

        if not rows:
            raise ValueError("No negative samples generated. Check dataset size.")

        self.user_idx = torch.tensor([r[0] for r in rows], dtype=torch.long)
        self.positive_song_idx = torch.tensor([r[1] for r in rows], dtype=torch.long)
        self.negative_song_idx = torch.tensor([r[2] for r in rows], dtype=torch.long)
        self.confidence = torch.tensor([r[3] for r in rows], dtype=torch.float32)

    def __len__(self) -> int:
        return int(len(self.user_idx))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.user_idx[index],
            self.positive_song_idx[index],
            self.negative_song_idx[index],
            self.confidence[index],
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _serializable_config(config: TrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


def _regularization_loss(model: HybridMusicModel) -> torch.Tensor:
    regularized = [
        model.user_embedding.weight,
        model.song_embedding.weight,
        model.user_bias.weight,
        model.song_bias.weight,
    ]
    return sum(torch.sum(p.pow(2)) for p in regularized) / max(1, model.config.num_users + model.config.num_songs)


def _batch_loss(
    model: HybridMusicModel,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tensors: dict[str, torch.Tensor],
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    user_idx, positive_song_idx, negative_song_idx, confidence = [item.to(device) for item in batch]

    user_features = tensors["user_features"][user_idx]
    positive_metadata = tensors["song_metadata"][positive_song_idx]
    negative_metadata = tensors["song_metadata"][negative_song_idx]
    positive_lyrics = tensors["lyric_embeddings"][positive_song_idx]
    negative_lyrics = tensors["lyric_embeddings"][negative_song_idx]

    positive_logits = model(user_idx, positive_song_idx, user_features, positive_metadata, positive_lyrics)
    negative_logits = model(user_idx, negative_song_idx, user_features, negative_metadata, negative_lyrics)

    # ✅ BPR loss (primary)
    rank_loss = -(confidence * F.logsigmoid(positive_logits - negative_logits)).mean()

    # ✅ BCE loss (auxiliary, lower weight)
    point_logits = torch.cat([positive_logits, negative_logits], dim=0)
    point_labels = torch.cat([torch.ones_like(positive_logits), torch.zeros_like(negative_logits)], dim=0)
    point_weights = torch.cat([confidence, torch.ones_like(confidence)], dim=0)
    point_loss = F.binary_cross_entropy_with_logits(
        point_logits, point_labels, weight=point_weights, reduction="mean"
    )

    reg_loss = _regularization_loss(model) if config.l2_reg > 0 else torch.zeros((), device=device)
    total_loss = config.lambda_rank * rank_loss + config.lambda_point * point_loss + config.l2_reg * reg_loss

    return {
        "total": total_loss,
        "point": point_loss.detach(),
        "rank": rank_loss.detach(),
        "reg": reg_loss.detach(),
    }


def _run_epoch(
    model: HybridMusicModel,
    dataloader: DataLoader,
    tensors: dict[str, torch.Tensor],
    config: TrainingConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    totals = {"total": 0.0, "point": 0.0, "rank": 0.0, "reg": 0.0}
    examples = 0

    for batch in dataloader:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        losses = _batch_loss(model, batch, tensors, config, device)
        if optimizer is not None:
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
            optimizer.step()

        batch_size = int(batch[0].shape[0])
        examples += batch_size
        for key in totals:
            totals[key] += float(losses[key].detach().cpu()) * batch_size

    return {key: value / max(1, examples) for key, value in totals.items()}


def _tensor_pack(prepared: PreparedHybridData, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "user_features": torch.from_numpy(prepared.user_features).to(device).float(),
        "song_metadata": torch.from_numpy(prepared.song_metadata).to(device).float(),
        "lyric_embeddings": torch.from_numpy(prepared.lyric_embeddings).to(device).float(),
    }


def _ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    gains = [1.0 if item in relevant else 0.0 for item in recommended[:k]]
    dcg = sum(gain / np.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    ideal_dcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    return float(dcg / ideal_dcg)


def _ranking_metrics_for_model(
    model: HybridMusicModel,
    tensors: dict[str, torch.Tensor],
    train_interactions: pd.DataFrame,
    validation_interactions: pd.DataFrame,
    k: int,
    device: torch.device,
    max_users: int = 50,
    batch_size: int = 4096,
    n_candidates: int = 100,             # ✅ SAMPLED eval — critical fix
) -> dict[str, float]:
    """
    CRITICAL FIX: sampled candidate evaluation.
    Full-catalog ranking over 80k songs at train time is:
      (a) extremely slow
      (b) produces near-zero recall because model hasn't learned to
          rank the true positive above 79,999 candidates yet.
    Sampled eval (100–1000 candidates) gives a meaningful training signal
    and matches the evaluation protocol used in standard RecSys benchmarks
    (NCF, LightGCN, etc.).
    """
    if validation_interactions.empty:
        return {"recall": 0.0, "ndcg": 0.0, "users": 0.0}

    rng_eval = np.random.default_rng(999)
    train_lookup = build_positive_lookup(train_interactions)
    validation_lookup = build_positive_lookup(validation_interactions)
    user_ids = sorted(validation_lookup)
    if max_users > 0:
        user_ids = user_ids[:max_users]

    all_song_indices = list(range(model.config.num_songs))
    recalls: list[float] = []
    ndcgs: list[float] = []
    model.eval()

    with torch.no_grad():
        for user_idx in user_ids:
            relevant = validation_lookup.get(user_idx, set())
            if not relevant:
                continue

            # ── SAMPLED CANDIDATE POOL ──────────────────────
            train_pos = train_lookup.get(user_idx, set())
            all_pos = relevant | train_pos
            neg_pool = [i for i in all_song_indices if i not in all_pos]
            n_neg = max(n_candidates - len(relevant), 0)
            sampled_neg = list(rng_eval.choice(neg_pool, size=min(n_neg, len(neg_pool)), replace=False))
            candidates = list(relevant) + sampled_neg  # positives always included
            candidates_tensor = torch.tensor(candidates, dtype=torch.long, device=device)

            # ── SCORE CANDIDATES ONLY ───────────────────────
            score_parts: list[np.ndarray] = []
            for start in range(0, len(candidates), batch_size):
                end = min(start + batch_size, len(candidates))
                song_ids_batch = candidates_tensor[start:end]
                user_ids_tensor = torch.full_like(song_ids_batch, int(user_idx))
                user_feats = tensors["user_features"][int(user_idx)].unsqueeze(0).expand(end - start, -1)
                logits = model(
                    user_ids_tensor,
                    song_ids_batch,
                    user_feats,
                    tensors["song_metadata"][song_ids_batch],
                    tensors["lyric_embeddings"][song_ids_batch],
                )
                score_parts.append(logits.detach().cpu().numpy())

            scores = np.concatenate(score_parts)
            ranked_local = np.argsort(-scores)
            ranked_song_ids = [candidates[i] for i in ranked_local]

            hits = len(set(ranked_song_ids[:k]) & relevant)
            recalls.append(hits / len(relevant))
            ndcgs.append(_ndcg_at_k(ranked_song_ids, relevant, k))

    return {
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "users": float(len(recalls)),
    }


def train_hybrid_model(
    prepared: PreparedHybridData,
    config: TrainingConfig,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    metrics_path: Path | str = DEFAULT_METRICS_PATH,
) -> tuple[HybridMusicModel, dict[str, Any]]:
    """Train and save the hybrid recommender."""
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    positives_to_avoid = build_positive_lookup(prepared.interactions)

    # ── Extract song genres for hard negative sampling ──────────
    song_genres: list[str] | None = None
    if hasattr(prepared, "song_catalog") and prepared.song_catalog is not None:
        catalog = prepared.song_catalog
        genre_col = next((c for c in ["genre", "Genre", "tag"] if c in catalog.columns), None)
        if genre_col:
            song_genres = catalog.sort_values("song_idx")[genre_col].fillna("unknown").astype(str).tolist()

    train_dataset = PairwiseInteractionDataset(
        positive_interactions=prepared.train_interactions,
        num_songs=len(prepared.index_to_song_id),
        positives_to_avoid=positives_to_avoid,
        negatives_per_positive=config.negatives_per_positive,
        hard_negative_ratio=config.hard_negative_ratio,
        lyric_embeddings=prepared.lyric_embeddings,
        song_genres=song_genres,
        seed=config.seed,
    )
    validation_dataset = PairwiseInteractionDataset(
        positive_interactions=(
            prepared.validation_interactions
            if not prepared.validation_interactions.empty
            else prepared.train_interactions
        ),
        num_songs=len(prepared.index_to_song_id),
        positives_to_avoid=positives_to_avoid,
        negatives_per_positive=config.negatives_per_positive,
        hard_negative_ratio=config.hard_negative_ratio,
        lyric_embeddings=prepared.lyric_embeddings,
        song_genres=song_genres,
        seed=config.seed + 10_000,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    model_config = HybridModelConfig(
        num_users=len(prepared.index_to_user_id),
        num_songs=len(prepared.index_to_song_id),
        user_feature_dim=int(prepared.user_features.shape[1]),
        metadata_dim=int(prepared.song_metadata.shape[1]),
        lyric_embedding_dim=int(prepared.lyric_embeddings.shape[1]),
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        interaction_type=config.interaction_type,
    )
    model = HybridMusicModel(model_config).to(device)
    tensors = _tensor_pack(prepared, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    # ✅ LR scheduler: reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-5
    )

    best_validation_loss = float("inf")
    best_validation_ndcg = -1.0
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        train_losses = _run_epoch(model, train_loader, tensors, config, device, optimizer=optimizer)
        with torch.no_grad():
            validation_losses = _run_epoch(model, validation_loader, tensors, config, device, optimizer=None)
            ranking_metrics = _ranking_metrics_for_model(
                model=model,
                tensors=tensors,
                train_interactions=prepared.train_interactions,
                validation_interactions=prepared.validation_interactions,
                k=config.ranking_eval_k,
                device=device,
                max_users=config.max_ranking_users,
                batch_size=max(1024, config.batch_size * 4),
                n_candidates=config.ranking_eval_candidates,
            )

        scheduler.step(ranking_metrics["ndcg"])

        should_update_best = ranking_metrics["ndcg"] > best_validation_ndcg
        if ranking_metrics["ndcg"] == best_validation_ndcg and validation_losses["total"] < best_validation_loss:
            should_update_best = True
        if should_update_best:
            best_validation_loss = validation_losses["total"]
            best_validation_ndcg = ranking_metrics["ndcg"]
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        record = {
            "epoch": float(epoch),
            "train_total": round(train_losses["total"], 6),
            "train_bce": round(train_losses["point"], 6),
            "train_bpr": round(train_losses["rank"], 6),
            "validation_total": round(validation_losses["total"], 6),
            "validation_bce": round(validation_losses["point"], 6),
            "validation_bpr": round(validation_losses["rank"], 6),
            f"validation_recall_at_{config.ranking_eval_k}": round(ranking_metrics["recall"], 6),
            f"validation_ndcg_at_{config.ranking_eval_k}": round(ranking_metrics["ndcg"], 6),
        }
        history.append(record)
        print(
            f"epoch={epoch:02d} "
            f"train_total={train_losses['total']:.4f} "
            f"train_bpr={train_losses['rank']:.4f} "
            f"val_recall@{config.ranking_eval_k}={ranking_metrics['recall']:.4f} "
            f"val_ndcg@{config.ranking_eval_k}={ranking_metrics['ndcg']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    artifact_paths = save_prepared_hybrid_data(prepared, config.output_dir)
    metrics = {
        "training_config": _serializable_config(config),
        "model_config": asdict(model_config),
        "num_users": int(model_config.num_users),
        "num_songs": int(model_config.num_songs),
        "train_positives": int(len(prepared.train_interactions)),
        "validation_positives": int(len(prepared.validation_interactions)),
        "train_pairs": int(len(train_dataset)),
        "validation_pairs": int(len(validation_dataset)),
        "best_epoch": int(best_epoch),
        "best_validation_loss": round(float(best_validation_loss), 6),
        f"best_validation_ndcg_at_{config.ranking_eval_k}": round(float(best_validation_ndcg), 6),
        "history": history,
        "artifact_paths": artifact_paths,
    }

    save_hybrid_checkpoint(
        model.cpu(),
        model_path,
        extra={
            "training_metrics": metrics,
            "data_artifacts": artifact_paths,
            "index_to_user_id": prepared.index_to_user_id,
            "index_to_song_id": prepared.index_to_song_id,
            "user_id_to_index": prepared.user_id_to_index,
            "song_id_to_index": prepared.song_id_to_index,
        },
    )
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return model.cpu(), metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hybrid music recommender.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_HYBRID_DATA_DIR)
    parser.add_argument("--interactions-path", type=Path, default=None)
    parser.add_argument("--user-features-path", type=Path, default=None)
    parser.add_argument("--song-characteristics-path", type=Path, default=None)
    parser.add_argument("--lyric-embeddings-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--latent-dim", type=int, default=TrainingConfig.latent_dim)
    parser.add_argument("--hidden-dim", type=int, default=TrainingConfig.hidden_dim)
    parser.add_argument("--dropout", type=float, default=TrainingConfig.dropout)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainingConfig.weight_decay)
    parser.add_argument("--negatives-per-positive", type=int, default=TrainingConfig.negatives_per_positive)
    parser.add_argument("--hard-negative-ratio", type=float, default=TrainingConfig.hard_negative_ratio)
    parser.add_argument("--lambda-point", type=float, default=TrainingConfig.lambda_point)
    parser.add_argument("--lambda-rank", type=float, default=TrainingConfig.lambda_rank)
    parser.add_argument("--l2-reg", type=float, default=TrainingConfig.l2_reg)
    parser.add_argument("--gradient-clip-norm", type=float, default=TrainingConfig.gradient_clip_norm)
    parser.add_argument("--confidence-alpha", type=float, default=TrainingConfig.confidence_alpha)
    parser.add_argument("--validation-ratio", type=float, default=TrainingConfig.validation_ratio)
    parser.add_argument("--min-play-count", type=float, default=TrainingConfig.min_play_count)
    parser.add_argument("--max-categories", type=int, default=TrainingConfig.max_categories)
    parser.add_argument("--interaction-type", choices=["dot", "mlp"], default=TrainingConfig.interaction_type)
    parser.add_argument("--ranking-eval-k", type=int, default=TrainingConfig.ranking_eval_k)
    parser.add_argument("--ranking-eval-candidates", type=int, default=TrainingConfig.ranking_eval_candidates)
    parser.add_argument("--max-ranking-users", type=int, default=TrainingConfig.max_ranking_users)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        data_dir=str(args.data_dir),
        output_dir=str(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        negatives_per_positive=args.negatives_per_positive,
        hard_negative_ratio=args.hard_negative_ratio,
        lambda_point=args.lambda_point,
        lambda_rank=args.lambda_rank,
        l2_reg=args.l2_reg,
        gradient_clip_norm=args.gradient_clip_norm,
        confidence_alpha=args.confidence_alpha,
        validation_ratio=args.validation_ratio,
        min_play_count=args.min_play_count,
        max_categories=args.max_categories,
        interaction_type=args.interaction_type,
        ranking_eval_k=args.ranking_eval_k,
        ranking_eval_candidates=args.ranking_eval_candidates,
        max_ranking_users=args.max_ranking_users,
        seed=args.seed,
    )

    raw = load_hybrid_csvs(
        data_dir=args.data_dir,
        interactions_path=args.interactions_path,
        user_features_path=args.user_features_path,
        song_characteristics_path=args.song_characteristics_path,
        lyric_embeddings_path=args.lyric_embeddings_path,
    )
    prepared = prepare_hybrid_data(
        interactions=raw.interactions,
        user_features=raw.user_features,
        song_characteristics=raw.song_characteristics,
        lyric_embeddings=raw.lyric_embeddings,
        min_play_count=config.min_play_count,
        confidence_alpha=config.confidence_alpha,
        validation_ratio=config.validation_ratio,
        seed=config.seed,
        max_categories=config.max_categories,
    )
    print(
        "loaded hybrid dataset: "
        f"users={len(prepared.index_to_user_id)} songs={len(prepared.index_to_song_id)} "
        f"train_positives={len(prepared.train_interactions)} "
        f"validation_positives={len(prepared.validation_interactions)}"
    )
    _, metrics = train_hybrid_model(
        prepared=prepared,
        config=config,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
    )
    print(f"saved model to {args.model_path}")
    print(f"saved metrics to {args.metrics_path}")
    print(f"best epoch: {metrics['best_epoch']}")
    print(f"best validation ndcg@{config.ranking_eval_k}: {metrics.get(f'best_validation_ndcg_at_{config.ranking_eval_k}', 'N/A')}")


if __name__ == "__main__":
    main()
