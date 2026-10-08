# 🎵 LyricLens

### Hybrid Music Recommendation System using Collaborative Filtering, Content Features & Semantic Lyric Embeddings

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-Application-ff4b4b?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/NLP-Lyric%20Embeddings-green" alt="NLP">
  <img src="https://img.shields.io/badge/Recommendation-Hybrid-purple" alt="Hybrid Recommendation">
</p>

**LyricLens** is a hybrid music recommendation system that combines collaborative signals, user information, song characteristics, and semantic lyric embeddings to generate personalized music recommendations.

Instead of relying only on genres or listening history, LyricLens incorporates the **semantic content of lyrics** alongside behavioral and metadata signals to build richer user and song representations.

---

## ✨ Key Features

- 🎧 **Hybrid Recommendation** — combines collaborative and content-based signals
- 👤 **User Representation** — trainable user embeddings with additional user features
- 🎵 **Song Representation** — combines song embeddings, metadata, and lyric representations
- 📝 **Semantic Lyric Features** — incorporates numerical lyric embeddings into recommendation
- 🧠 **Gated Feature Fusion** — learns the contribution of different song representations
- 📈 **Implicit Feedback Learning** — learns from user-song interaction and play-count data
- ⚖️ **BCE + BPR Training Objective** — combines pointwise prediction with pairwise ranking
- 📊 **Ranking Evaluation** — supports Recall@K and NDCG@K
- 🖥️ **Interactive Interface** — Streamlit-based application for recommendation workflows
- 🧩 **Modular Architecture** — separate pipelines for data processing, training, evaluation, and inference

---

## 🧠 System Architecture

LyricLens learns representations for both users and songs before estimating their compatibility.

```text
                         LyricLens
                            │
             ┌──────────────┴──────────────┐
             │                             │
         USER SIDE                     SONG SIDE
             │                             │
      ┌──────┴──────┐          ┌───────────┼───────────┐
      │             │          │           │           │
   User ID      User Features Song ID    Metadata    Lyrics
      │             │          │           │           │
      ▼             ▼          ▼           ▼           ▼
  Trainable        MLP      Trainable     MLP         MLP
  Embedding       Encoder   Embedding    Encoder      Encoder
      │             │          │           │           │
      └──────┬──────┘          └───────────┼───────────┘
             │                         Gated Fusion
             ▼                             │
        User Vector                   Song Vector
             │                             │
             └──────────────┬──────────────┘
                            ▼
                   Interaction Function
                            │
                            ▼
                  Recommendation Score
```

### User Representation

A learned user embedding is combined with encoded user features:

```text
User ID → Trainable Embedding ──┐
                                ├── LayerNorm → User Vector
User Features → MLP Encoder ────┘
```

### Song Representation

Each song can be represented through three complementary sources:

```text
Song ID              → Trainable Song Embedding
Song Characteristics → Metadata MLP Encoder
Lyric Embeddings     → Lyric MLP Encoder
                              │
                              ▼
                         Gated Fusion
                              │
                              ▼
                         Song Vector
```

The gating mechanism learns how strongly each representation should contribute to the final song representation.

---

## ⚙️ Recommendation & Training

Listening behavior is treated as **implicit feedback**.

```text
play_count > 0  →  positive preference
```

Interaction confidence is derived from listening frequency:

```text
confidence = 1 + α × log(1 + play_count)
```

The training objective combines binary classification and pairwise ranking:

```text
Loss = λ_point × BCEWithLogits
     + λ_rank  × Confidence-Weighted BPR
     + Regularization
```

This enables the model to learn both:

- whether a user is likely to interact with a song, and
- how relevant that song should be compared with alternatives.

---

## 📊 Evaluation

LyricLens uses ranking-oriented metrics suitable for recommendation systems.

| Metric | Purpose |
|---|---|
| **Recall@K** | Measures how many relevant songs appear within the top-K recommendations |
| **NDCG@K** | Measures ranking quality while rewarding relevant songs placed higher |

Training positives are masked during validation ranking to avoid information leakage.

Run evaluation with:

```bash
python evaluate.py --k 5 10 20
```

---

## 📁 Project Structure

```text
LyricLens/
│
├── app.py                       # Streamlit application
├── recommender.py               # Recommendation integration
├── evaluate.py                  # Ranking evaluation
├── requirements.txt
│
├── src/
│   ├── data/
│   │   └── data_loader.py
│   │
│   ├── features/
│   │   ├── embedding.py
│   │   └── emotion_theme.py
│   │
│   ├── models/
│   │   ├── collaborative.py
│   │   ├── hybrid_model.py
│   │   └── model.py
│   │
│   ├── recommender/
│   │   └── recommender.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   └── trainer.py
│   │
│   └── evaluation/
│       ├── evaluator.py
│       └── preprocess.py
│
├── scripts/
│   ├── create_sample_dataset.py
│   ├── import_external_datasets.py
│   └── regenerate_data.py
│
├── notebooks/                   # Experiments and model development
├── artifacts/                   # Metrics and lightweight model metadata
└── data/                        # Dataset metadata
```

Large datasets, virtual environments, generated feature stores, and trained model weights are excluded from version control.

---

## 📦 Dataset Format

The hybrid training pipeline works with four primary data sources:

| Dataset | Contents |
|---|---|
| `interactions.csv` | `user_id`, `song_id`, `play_count` |
| `user_features.csv` | User-level numerical/categorical features |
| `song_characteristics.csv` | Song metadata such as genre, tempo and energy |
| `lyric_embeddings.csv` | Numerical semantic representations of song lyrics |

The preprocessing pipeline performs ID mapping, feature encoding, interaction filtering, and train/validation preparation.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/SuvratJoshi2275/LyricLens.git
cd LyricLens
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Model Training

Train the hybrid recommendation model:

```bash
python src/training/train.py --epochs 8
```

For a custom dataset:

```bash
python src/training/train.py \
  --data-dir /path/to/music_dataset \
  --epochs 20 \
  --negatives-per-positive 6
```

Training generates model checkpoints, feature stores, interaction splits, metadata, and training metrics.

---

## 🎧 Run LyricLens

Launch the Streamlit application:

```bash
streamlit run app.py
```

The interface provides access to the recommendation workflow and hybrid scoring functionality.

---

## 🔬 Experiments

The `notebooks/` directory contains experimentation and development work covering:

- music recommendation pipelines
- hybrid model training
- semantic recommendation experiments
- custom dataset training
- recommendation demonstrations

Core reusable functionality is organized separately under `src/`.

---

## 🛠️ Tech Stack

| Area | Technologies |
|---|---|
| **Language** | Python |
| **Deep Learning** | PyTorch |
| **Machine Learning** | Scikit-learn |
| **NLP** | Transformers, semantic lyric embeddings |
| **Data Processing** | Pandas, NumPy |
| **Interface** | Streamlit |
| **Evaluation** | Recall@K, NDCG@K |
| **Version Control** | Git, GitHub |

---

## 🔮 Future Improvements

- Scale recommendation to larger interaction datasets
- Approximate nearest-neighbor retrieval for large song catalogs
- Richer transformer-based lyric representations
- Improved cold-start recommendations for new users and songs
- Online user-feedback integration
- Production API deployment

---

## 👤 Author

**Suvrat Joshi**  
Computer Science & Engineering — Artificial Intelligence & Machine Learning

---

<p align="center">
  <i>Exploring music beyond genres through behavior, content, and the semantic meaning of lyrics.</i>
</p>
