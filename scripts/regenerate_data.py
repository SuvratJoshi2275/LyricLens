import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

# ✅ FIXED PATH

DATA_DIR = Path("data")

# ─────────────────────────────────────────────

# 1. LOAD SONG CATALOG

# ─────────────────────────────────────────────

song_chars = pd.read_csv(DATA_DIR / "song_characteristics.csv")
song_chars["song_id"] = song_chars["song_id"].astype(str)
NUM_SONGS = len(song_chars)
song_ids = song_chars["song_id"].tolist()

# Detect genre column

genre_col = next((c for c in ["genre", "Genre", "tag"] if c in song_chars.columns), None)
genres_per_song = (
song_chars[genre_col].fillna("unknown").astype(str).tolist()
if genre_col else ["unknown"] * NUM_SONGS
)
unique_genres = list(set(genres_per_song))

# ─────────────────────────────────────────────

# 2. LOAD LYRIC EMBEDDINGS

# ─────────────────────────────────────────────

lyric_df = pd.read_csv(DATA_DIR / "lyric_embeddings.csv")
lyric_df["song_id"] = lyric_df["song_id"].astype(str)
emb_cols = [c for c in lyric_df.columns if c != "song_id"]

lyric_aligned = (
pd.DataFrame({"song_id": song_ids})
.merge(lyric_df, on="song_id", how="left")
.fillna(0.0)
)

emb_matrix = lyric_aligned[emb_cols].to_numpy(dtype=np.float32)
norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
emb_matrix = emb_matrix / np.maximum(norms, 1e-8)

# ─────────────────────────────────────────────

# 3. GENERATE INTERACTIONS

# ─────────────────────────────────────────────

NUM_USERS = 200
POSITIVES_PER_USER = 20
NEG_PER_POS = 20
HARD_NEG_RATIO = 0.40

interaction_rows = []
user_profile_rows = []

song_idx_by_genre = {}
for i, g in enumerate(genres_per_song):
song_idx_by_genre.setdefault(g, []).append(i)

for uid in range(NUM_USERS):
n_pref_genres = rng.integers(1, 3)
pref_genres = list(rng.choice(unique_genres, size=min(n_pref_genres, len(unique_genres)), replace=False))

```
pref_pool = []
for g in pref_genres:
    pref_pool.extend(song_idx_by_genre.get(g, []))
pref_pool = list(set(pref_pool))

if len(pref_pool) < POSITIVES_PER_USER:
    extra = rng.choice(NUM_SONGS, size=POSITIVES_PER_USER * 3, replace=False).tolist()
    pref_pool = list(set(pref_pool + extra))

anchor = int(rng.choice(pref_pool[:min(len(pref_pool), 50)]))
sim_scores = emb_matrix[pref_pool] @ emb_matrix[anchor]

temp = 0.5
exp_scores = np.exp((sim_scores - sim_scores.max()) / temp)
probs = exp_scores / exp_scores.sum()

chosen = rng.choice(len(pref_pool), size=POSITIVES_PER_USER, replace=False, p=probs)
pos_indices = [pref_pool[i] for i in chosen]

for song_idx in pos_indices:
    interaction_rows.append({
        "user_id": uid,
        "song_id": song_ids[song_idx],
        "interaction": 1,
        "play_count": int(1 + rng.poisson(3)),
    })

pos_set = set(pos_indices)
pos_genres = set(genres_per_song[i] for i in pos_indices)

for pos_idx in pos_indices:
    n_hard = int(NEG_PER_POS * HARD_NEG_RATIO)
    n_easy = NEG_PER_POS - n_hard

    same_genre_pool = [
        i for g in pos_genres
        for i in song_idx_by_genre.get(g, [])
        if i not in pos_set
    ]

    if len(same_genre_pool) >= n_hard:
        sim = emb_matrix[same_genre_pool] @ emb_matrix[pos_idx]
        top = np.argsort(-sim)[:n_hard * 3]
        chosen_hard = list(rng.choice([same_genre_pool[i] for i in top], size=n_hard, replace=False))
    else:
        chosen_hard = same_genre_pool[:n_hard]

    easy_pool = [i for i in range(NUM_SONGS) if i not in pos_set and i not in set(chosen_hard)]
    chosen_easy = list(rng.choice(easy_pool, size=n_easy, replace=False))

    for neg_idx in chosen_hard + chosen_easy:
        interaction_rows.append({
            "user_id": uid,
            "song_id": song_ids[neg_idx],
            "interaction": 0,
            "play_count": 0,
        })
```

interactions_df = pd.DataFrame(interaction_rows)
interactions_df.to_csv(DATA_DIR / "interactions.csv", index=False)

print("✅ interactions.csv generated:", len(interactions_df))

# ─────────────────────────────────────────────

# 4. USER FEATURES

# ─────────────────────────────────────────────

pos_df = interactions_df[interactions_df["interaction"] == 1]

merged = pos_df.merge(song_chars[["song_id", genre_col]], on="song_id", how="left")

user_rows = []
global_dist = pd.Series(genres_per_song).value_counts(normalize=True).to_dict()

for uid in range(NUM_USERS):
u = merged[merged["user_id"] == uid]

```
genre_aff = {}
if not u.empty:
    gc = u[genre_col].value_counts(normalize=True)
    for g in unique_genres[:10]:
        genre_aff[f"genre_affinity_{g}"] = float(gc.get(g, 0.0))

rel_pref = {}
for g in unique_genres[:10]:
    rel_pref[f"rel_pref_{g}"] = genre_aff.get(f"genre_affinity_{g}", 0.0) / max(global_dist.get(g, 1e-6), 1e-6)

probs = np.array(list(genre_aff.values()) or [1.0])
probs = probs / probs.sum()
entropy = float(-np.sum(probs * np.log(probs + 1e-9)))

user_rows.append({
    "user_id": uid,
    "n_positive": len(u),
    "genre_entropy": entropy,
    **genre_aff,
    **rel_pref
})
```

user_df = pd.DataFrame(user_rows).fillna(0.0)
user_df.to_csv(DATA_DIR / "user_features.csv", index=False)

print("✅ user_features.csv generated")
print("🔥 DONE — now run: python train.py")
