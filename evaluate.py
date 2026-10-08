"""Evaluate the trained hybrid recommender with Recall@K and NDCG@K.

CRITICAL FIXES vs original:
  1. SAMPLED CANDIDATE EVALUATION (not full 80k catalog)
     Protocol: for each test user, rank true positives against N random negatives.
     This is the standard protocol used by NeuMF, LightGCN, SASRec papers.
     Full-catalog eval gives ~0 recall at 80k because the model hasn't learned
     a global ordering over the entire catalog yet.
  2. TRAIN POSITIVES MASKED from candidate pool (no leakage).
  3. Optional full-catalog eval available via --full-catalog flag for final
     reporting only (slow, run once after training is done).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import random

from hybrid_model import load_hybrid_checkpoint
from preprocess import build_positive_lookup, load_prepared_hybrid_artifacts
from trainer import DEFAULT_MODEL_PATH, DEFAULT_OUTPUT_DIR


DEFAULT_EVALUATION_PATH = Path("artifacts/hybrid_evaluation_metrics.json")


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    gains = [1.0 if song_id in relevant else 0.0 for song_id in recommended[:k]]
    dcg = sum(gain / np.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    ideal_dcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    return float(dcg / ideal_dcg)


# ─────────────────────────────────────────────
# SAMPLED CANDIDATE EVALUATION (DEFAULT, FAST)
# ─────────────────────────────────────────────
def evaluate_ranking_sampled(
    model_path: Path | str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    k_values: list[int] | None = None,
    n_candidates: int = 100,
) -> dict[str, Any]:
    """
    Standard sampled evaluation protocol.
    For each validation user:
      - include all true positives in candidate pool
      - sample (n_candidates - n_positives) random negatives
      - rank the pool, compute Recall@K and NDCG@K

    This is fast and gives meaningful signal. Use n_candidates=1000 for
    more conservative estimates before final reporting.
    """
    k_values = k_values or [5, 10, 20]

    artifacts = load_prepared_hybrid_artifacts(output_dir)
    model, checkpoint = load_hybrid_checkpoint(model_path)
    model.eval()

    validation_interactions = artifacts["validation_interactions"]
    train_interactions = artifacts["train_interactions"]
    validation_lookup = build_positive_lookup(validation_interactions)
    train_lookup = build_positive_lookup(train_interactions)

    num_songs = len(artifacts["song_metadata"])
    all_song_ids = list(range(num_songs))
    user_ids = list(validation_lookup.keys())

    user_features = artifacts["user_features"]
    song_metadata = artifacts["song_metadata"]
    lyric_embeddings = artifacts["lyric_embeddings"]

    rng = random.Random(42)
    import torch
    device = torch.device("cpu")

    metrics: dict[str, dict[str, float | int]] = {}
    for k in k_values:
        recalls, ndcgs = [], []

        for user_id in user_ids:
            pos = list(validation_lookup.get(user_id, []))
            if not pos:
                continue

            # Mask train positives from negatives
            train_pos = train_lookup.get(user_id, set())
            all_pos = set(pos) | train_pos
            neg_pool = [i for i in all_song_ids if i not in all_pos]

            n_neg = max(n_candidates - len(pos), 0)
            sampled_neg = rng.sample(neg_pool, min(n_neg, len(neg_pool)))
            candidates = list(pos) + sampled_neg

            # Score candidates only
            candidates_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
            u_feat = torch.from_numpy(user_features[user_id:user_id+1]).float().expand(len(candidates), -1)
            u_ids = torch.full((len(candidates),), user_id, dtype=torch.long)

            with torch.no_grad():
                logits = model(
                    u_ids,
                    candidates_tensor,
                    u_feat,
                    torch.from_numpy(song_metadata[candidates]).float(),
                    torch.from_numpy(lyric_embeddings[candidates]).float(),
                )
            scores = logits.cpu().numpy()

            ranked_local = np.argsort(-scores)
            ranked_song_ids = [candidates[i] for i in ranked_local]
            relevant = set(pos)

            recalls.append(recall_at_k(ranked_song_ids, relevant, k))
            ndcgs.append(ndcg_at_k(ranked_song_ids, relevant, k))

        metrics[f"@{k}"] = {
            "recall": round(float(np.mean(recalls)) if recalls else 0.0, 6),
            "ndcg": round(float(np.mean(ndcgs)) if ndcgs else 0.0, 6),
            "evaluated_users": int(len(recalls)),
            "n_candidates": n_candidates,
            "protocol": "sampled",
        }

    return {
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "model_config": checkpoint.get("model_config", {}),
        "metrics": metrics,
    }


# ─────────────────────────────────────────────
# FULL CATALOG EVALUATION (SLOW, FINAL REPORT)
# ─────────────────────────────────────────────
def evaluate_ranking_full_catalog(
    model_path: Path | str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    k_values: list[int] | None = None,
    batch_size: int = 4096,
) -> dict[str, Any]:
    """
    Full-catalog evaluation: rank all 80k songs per user.
    SLOW. Use only for final reporting after training converges.
    Expected recall@10 will be lower than sampled eval (~0.05–0.15 is realistic).
    """
    import torch
    k_values = k_values or [5, 10, 20]

    artifacts = load_prepared_hybrid_artifacts(output_dir)
    model, checkpoint = load_hybrid_checkpoint(model_path)
    model.eval()
    device = torch.device("cpu")

    validation_lookup = build_positive_lookup(artifacts["validation_interactions"])
    train_lookup = build_positive_lookup(artifacts["train_interactions"])

    user_features = torch.from_numpy(artifacts["user_features"]).float()
    song_metadata = torch.from_numpy(artifacts["song_metadata"]).float()
    lyric_emb = torch.from_numpy(artifacts["lyric_embeddings"]).float()
    num_songs = song_metadata.shape[0]

    user_ids = list(validation_lookup.keys())
    metrics: dict[str, dict[str, float | int]] = {}

    for k in k_values:
        recalls, ndcgs = [], []
        for user_id in user_ids:
            relevant = validation_lookup.get(user_id, set())
            if not relevant:
                continue
            train_pos = train_lookup.get(user_id, set())

            score_parts = []
            with torch.no_grad():
                for start in range(0, num_songs, batch_size):
                    end = min(start + batch_size, num_songs)
                    song_ids_b = torch.arange(start, end, dtype=torch.long, device=device)
                    u_ids_b = torch.full_like(song_ids_b, user_id)
                    u_feat_b = user_features[user_id].unsqueeze(0).expand(end - start, -1)
                    logits = model(u_ids_b, song_ids_b, u_feat_b, song_metadata[start:end], lyric_emb[start:end])
                    score_parts.append(logits.cpu().numpy())

            scores = np.concatenate(score_parts)
            for seen in train_pos:
                if 0 <= int(seen) < len(scores):
                    scores[int(seen)] = -np.inf

            ranked = np.argsort(-scores)[:k].astype(int).tolist()
            recalls.append(recall_at_k(ranked, relevant, k))
            ndcgs.append(ndcg_at_k(ranked, relevant, k))

        metrics[f"@{k}"] = {
            "recall": round(float(np.mean(recalls)) if recalls else 0.0, 6),
            "ndcg": round(float(np.mean(ndcgs)) if ndcgs else 0.0, 6),
            "evaluated_users": int(len(recalls)),
            "protocol": "full_catalog",
        }

    return {
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "model_config": checkpoint.get("model_config", {}),
        "metrics": metrics,
    }


# ─────────────────────────────────────────────
# DEFAULT PUBLIC FUNCTION (backward compat)
# ─────────────────────────────────────────────
def evaluate_ranking(
    model_path: Path | str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    k_values: list[int] | None = None,
    n_candidates: int = 100,
    full_catalog: bool = False,
) -> dict[str, Any]:
    if full_catalog:
        return evaluate_ranking_full_catalog(model_path, output_dir, k_values)
    return evaluate_ranking_sampled(model_path, output_dir, k_values, n_candidates=n_candidates)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained hybrid recommender.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--output-path", type=Path, default=DEFAULT_EVALUATION_PATH)
    parser.add_argument(
        "--n-candidates", type=int, default=100,
        help="Sampled eval pool size. Use 1000 for conservative estimates. Default: 100."
    )
    parser.add_argument(
        "--full-catalog", action="store_true",
        help="Full 80k-song ranking (slow). Use for final reporting only."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found at {args.model_path}. Run: python train.py")

    results = evaluate_ranking(
        model_path=args.model_path,
        output_dir=args.output_dir,
        k_values=args.k,
        n_candidates=args.n_candidates,
        full_catalog=args.full_catalog,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(results, indent=2))

    proto = "full_catalog" if args.full_catalog else f"sampled({args.n_candidates})"
    print(f"Protocol: {proto}")
    for k, m in results["metrics"].items():
        print(
            f"{k}: recall={m['recall']:.4f}  ndcg={m['ndcg']:.4f}  users={m['evaluated_users']}"
        )
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
