import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os
import html as html_lib
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "training"
FINAL_SONGS_PATH = DATA_DIR / "final_songs_FIXED.csv"

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LyricLens",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── inject global CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Circular+Std:wght@400;700&family=DM+Sans:wght@300;400;500;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

/* ── reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #0f0f0f !important;
    color: #ffffff !important;
}
.stApp { background-color: #0f0f0f !important; }
section[data-testid="stSidebar"] {
    background: #0a0a0a !important;
    border-right: 1px solid #1a1a1a;
}
section[data-testid="stSidebar"] * { color: #ffffff !important; }

/* ── hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1400px; }

/* ── scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0f0f0f; }
::-webkit-scrollbar-thumb { background: #1DB954; border-radius: 3px; }

/* ── sidebar logo ── */
.sidebar-logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #1DB954 0%, #1ed760 50%, #4ade80 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding: 0.5rem 0 1.5rem 0;
    display: block;
}
.sidebar-nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; border-radius: 8px;
    cursor: pointer; transition: background 0.2s;
    font-size: 0.92rem; font-weight: 500;
    color: #b3b3b3 !important;
    text-decoration: none;
}
.sidebar-nav-item:hover { background: #1a1a1a; color: #fff !important; }
.sidebar-nav-item.active { background: #1a1a1a; color: #1DB954 !important; }

/* ── hero section ── */
.hero-container {
    background: linear-gradient(135deg, #0f0f0f 0%, #0d2818 40%, #0f0f0f 100%);
    border: 1px solid #1a2e1a;
    border-radius: 20px;
    padding: 3.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: "";
    position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 30% 60%, rgba(29,185,84,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.8rem; font-weight: 700;
    line-height: 1.1; letter-spacing: -2px;
    background: linear-gradient(135deg, #ffffff 0%, #1DB954 60%, #4ade80 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.75rem;
}
.hero-sub {
    font-size: 1.1rem; color: #a0a0a0;
    max-width: 480px; line-height: 1.6;
    margin-bottom: 1.5rem;
}

/* ── stat cards ── */
.stat-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.5rem 0; }
.stat-card {
    background: #181818;
    border: 1px solid #282828;
    border-radius: 14px;
    padding: 1.25rem 1.75rem;
    min-width: 160px;
    flex: 1;
}
.stat-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem; font-weight: 700;
    color: #1DB954; line-height: 1;
}
.stat-label {
    font-size: 0.78rem; color: #6a6a6a;
    text-transform: uppercase; letter-spacing: 1px;
    margin-top: 0.3rem;
}

/* ── song cards ── */
.song-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}
.song-card {
    background: #181818;
    border: 1px solid #242424;
    border-radius: 14px;
    padding: 1.25rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    position: relative;
    overflow: hidden;
    cursor: pointer;
}
.song-card:hover {
    transform: translateY(-4px) scale(1.01);
    box-shadow: 0 16px 40px rgba(0,0,0,0.5), 0 0 0 1px #1DB954;
    border-color: #1DB954;
    background: #1c1c1c;
}
.song-card::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #1DB954, #4ade80);
    opacity: 0;
    transition: opacity 0.2s;
}
.song-card:hover::before { opacity: 1; }
.song-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #1DB954, #0d7a35);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    margin-bottom: 0.85rem;
}
.song-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem; font-weight: 600;
    color: #fff;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    margin-bottom: 0.25rem;
}
.song-artist {
    font-size: 0.8rem; color: #8a8a8a;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    margin-bottom: 0.4rem;
}
.song-meta {
    font-size: 0.7rem; color: #1DB954;
    font-weight: 600; letter-spacing: 0.3px;
    margin-bottom: 0.6rem;
}
.song-lyric {
    font-size: 0.76rem; color: #5a5a5a;
    line-height: 1.45;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    margin-bottom: 0.85rem;
}
.sim-bar-wrap {
    background: #282828;
    border-radius: 99px;
    height: 4px;
    overflow: hidden;
    margin-top: 0.3rem;
}
.sim-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #1DB954, #4ade80);
}
.sim-label {
    font-size: 0.72rem; color: #1DB954;
    font-weight: 600; margin-bottom: 2px;
    font-family: 'Space Grotesk', sans-serif;
}

/* ── section header ── */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem; font-weight: 700;
    color: #fff; margin: 1.8rem 0 0.8rem;
    letter-spacing: -0.5px;
}
.section-sub {
    font-size: 0.85rem; color: #6a6a6a;
    margin-bottom: 1.2rem;
}

/* ── search bar ── */
.stTextInput > div > div > input {
    background: #181818 !important;
    border: 1px solid #333 !important;
    border-radius: 50px !important;
    color: #fff !important;
    padding: 0.75rem 1.25rem !important;
    font-size: 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #1DB954 !important;
    box-shadow: 0 0 0 2px rgba(29,185,84,0.2) !important;
}

/* ── select box ── */
.stSelectbox > div > div {
    background: #181818 !important;
    border: 1px solid #333 !important;
    border-radius: 10px !important;
    color: #fff !important;
}

/* ── buttons ── */
.stButton > button {
    background: #1DB954 !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.6rem 1.8rem !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 0 0 rgba(29,185,84,0) !important;
}
.stButton > button:hover {
    background: #1ed760 !important;
    box-shadow: 0 0 20px rgba(29,185,84,0.4) !important;
    transform: scale(1.03) !important;
}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0.5rem !important;
    border-bottom: 1px solid #1a1a1a;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8a8a8a !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 0.5rem 1rem !important;
    font-size: 0.88rem !important;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #1DB954 !important;
    border-bottom: 2px solid #1DB954 !important;
}

/* ── divider ── */
hr { border-color: #1a1a1a !important; }

/* ── metric ── */
[data-testid="stMetricValue"] {
    color: #1DB954 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.8rem !important;
}
[data-testid="stMetricLabel"] { color: #8a8a8a !important; }

/* ── history badge ── */
.hist-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #181818; border: 1px solid #282828;
    border-radius: 50px; padding: 5px 12px;
    font-size: 0.8rem; color: #ccc; margin: 4px;
}
.hist-badge span.dot { color: #1DB954; font-size: 0.7rem; }

/* ── trending chip ── */
.trend-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: #181818; border: 1px solid #1DB954;
    border-radius: 50px; padding: 4px 14px;
    font-size: 0.78rem; color: #1DB954;
    font-weight: 600; margin: 4px;
    font-family: 'Space Grotesk', sans-serif;
}

/* ── matplotlib dark ── */
.stImage img { border-radius: 14px; }

/* ── pagination ── */
.page-pill {
    display: inline-block; padding: 5px 14px;
    background: #181818; border: 1px solid #333;
    border-radius: 50px; color: #ccc;
    font-size: 0.82rem; margin: 2px;
}
.page-pill.active {
    background: #1DB954; color: #000;
    font-weight: 700; border-color: #1DB954;
}

/* ── no results ── */
.no-results {
    text-align: center; padding: 4rem 1rem;
    color: #5a5a5a; font-size: 1rem;
}
.no-results .icon { font-size: 3rem; margin-bottom: 0.75rem; }

/* ── profile header ── */
.profile-header {
    background: linear-gradient(135deg, #0a1f0a, #0f0f0f);
    border: 1px solid #1a2e1a;
    border-radius: 18px;
    padding: 2.5rem;
    display: flex; align-items: center; gap: 1.5rem;
    margin-bottom: 1.5rem;
}
.profile-avatar {
    width: 72px; height: 72px;
    background: linear-gradient(135deg, #1DB954, #0d4a22);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem; flex-shrink: 0;
}
.profile-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem; font-weight: 700;
    color: #fff;
}
.profile-sub { font-size: 0.85rem; color: #6a6a6a; margin-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)


# ── session state init ──────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "recent_searches" not in st.session_state:
    st.session_state.recent_searches = []
if "explorer_page" not in st.session_state:
    st.session_state.explorer_page = 0


# ── data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    if not FINAL_SONGS_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{FINAL_SONGS_PATH}\n"
            f"Expected final_songs_FIXED.csv inside the project's data/ directory."
        )
    df = pd.read_csv(FINAL_SONGS_PATH)
    return df


@st.cache_resource(show_spinner=False)
def get_recommender(df):
    sys.path.insert(0, str(PROJECT_DIR))
    from recommender import LyricRecommender, HybridRecommender, HybridUnavailable

    try:
        return HybridRecommender(df, MODEL_DIR)
    except HybridUnavailable as e:
        rec = LyricRecommender(df)
        rec.fallback_reason = str(e)
        return rec


try:
    df = load_data()
    rec = get_recommender(df)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    meta_cols = [c for c in df.columns if not c.startswith("emb_")]
    DATA_OK = True
except Exception as e:
    DATA_OK = False
    DATA_ERR = str(e)


# ── helpers ────────────────────────────────────────────────────────────────────
def strip_brackets(val, limit=60):
    """Plain-text artist/field cleanup, e.g. '["Elvis Presley"]' -> 'Elvis Presley'
    and '["A", "B"]' -> 'A, B'. Parses as an actual Python list literal (the
    stored format) rather than naive slicing, so multi-artist entries don't
    end up with stray quotes/commas. No HTML escaping — use for st widget
    labels and matplotlib text."""
    s = str(val).strip()
    if s in ("nan", "None", ""):
        return ""
    if s.startswith("[") and s.endswith("]"):
        try:
            import ast
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                s = ", ".join(str(x) for x in parsed)
        except (ValueError, SyntaxError):
            s = s[1:-1].strip().strip("'\"")
    return s[:limit]


def clean_field(val, limit=60):
    """Same bracket/quote strip, HTML-escaped — use inside unsafe_allow_html markup."""
    return html_lib.escape(strip_brackets(val, limit))


def song_card_html(title, artist, lyric="", sim=None, genre=None, year=None):
    safe_title = clean_field(title, 48)
    safe_artist = clean_field(artist, 36)
    raw_lyric = str(lyric).strip()
    lyric_preview = html_lib.escape(raw_lyric[:80]) if raw_lyric not in ("nan", "None", "") else ""

    meta_bits = []
    g = str(genre).strip() if genre is not None else ""
    if g not in ("", "nan", "None"):
        meta_bits.append(clean_field(g, 24))
    y = str(year).strip() if year is not None else ""
    if y not in ("", "nan", "None"):
        try:
            y = str(int(float(y)))
        except ValueError:
            pass
        meta_bits.append(html_lib.escape(y))
    meta_section = f'<div class="song-meta">{" · ".join(meta_bits)}</div>' if meta_bits else ""

    sim_section = ""
    if sim is not None:
        pct = max(0, min(100, int(float(sim) * 100)))
        width = max(4, pct)
        sim_section = (
            f'<div class="sim-label">{pct}% match</div>'
            f'<div class="sim-bar-wrap"><div class="sim-bar-fill" style="width:{width}%"></div></div>'
        )

    lyric_section = f'<div class="song-lyric">{lyric_preview}</div>' if lyric_preview else ""

    return (
        f'<div class="song-card">'
        f'<div class="song-icon">🎵</div>'
        f'<div class="song-title">{safe_title}</div>'
        f'<div class="song-artist">{safe_artist}</div>'
        f'{meta_section}'
        f'{lyric_section}'
        f'{sim_section}'
        f'</div>'
    )


def render_song_grid(rows, lyric_col=None, sim_col=None, genre_col=None, year_col=None, cols=4):
    all_html = '<div class="song-grid">'
    for _, row in rows.iterrows():
        lyric = row.get(lyric_col, "") if lyric_col else ""
        sim = row.get(sim_col) if sim_col else None
        genre = row.get(genre_col) if genre_col else None
        year = row.get(year_col) if year_col else None
        all_html += song_card_html(
            row.get("song_name", "Unknown"),
            row.get("artist", "Unknown"),
            lyric=lyric, sim=sim, genre=genre, year=year,
        )
    all_html += "</div>"
    st.markdown(all_html, unsafe_allow_html=True)


def detect_col(candidates):
    for c in df.columns:
        if any(k in c.lower() for k in candidates):
            return c
    return None


def add_recent(song_name, artist):
    entry = f"{song_name} — {artist}"
    lst = st.session_state.recent_searches
    if entry not in lst:
        lst.insert(0, entry)
    st.session_state.recent_searches = lst[:10]


# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="sidebar-logo">🎵 LyricLens</span>', unsafe_allow_html=True)
    st.markdown("---")

    nav_items = [
        ("🏠", "Home"),
        ("🔍", "Discover"),
        ("🗂️", "Explorer"),
        ("📊", "Analytics"),
        ("👤", "Profile"),
    ]
    for icon, label in nav_items:
        active_class = "active" if st.session_state.page == label else ""
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label
            st.rerun()

    st.markdown("---")
    if DATA_OK:
        st.markdown(
            f'<div style="font-size:0.75rem;color:#555;padding:0 8px">'
            f'<b style="color:#1DB954">{len(df):,}</b> songs indexed<br>'
            f'<b style="color:#1DB954">{len(emb_cols)}</b> embedding dims'
            f'</div>',
            unsafe_allow_html=True,
        )


page = st.session_state.page

if DATA_OK and getattr(rec, "backend", "cosine") != "hybrid":
    st.error(
        f"Hybrid model failed to load: {getattr(rec, 'fallback_reason', 'unknown reason')}\n\n"
        f"Running on the cosine-only fallback recommender instead. Fix the error above before relying on results."
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 · HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "Home":
    if not DATA_OK:
        st.error(f"Failed to load dataset: {DATA_ERR}")
        st.stop()

    st.markdown("""
    <div class="hero-container">
      <div class="hero-title">Discover Music<br>Through Lyrics.</div>
      <div class="hero-sub">
        Semantic similarity powered by 384-dimensional lyric embeddings.
        Find songs that sound like what you love.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # stats
    n_songs = len(df)
    n_artists = df["artist"].nunique() if "artist" in df.columns else 0
    n_dims = len(emb_cols)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-value">{n_songs:,}</div>
          <div class="stat-label">Total Songs</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-value">{n_artists:,}</div>
          <div class="stat-label">Unique Artists</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-value">{n_dims}</div>
          <div class="stat-label">Embedding Dims</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">🔥 Trending Right Now</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Randomly sampled highlights from the dataset</div>', unsafe_allow_html=True)

    sample = df.sample(min(12, len(df)), random_state=42)
    lyric_col = next((c for c in df.columns if "lyric" in c.lower()), None)
    genre_col = detect_col(["genre"])
    year_col = detect_col(["year"])
    render_song_grid(sample, lyric_col=lyric_col, genre_col=genre_col, year_col=year_col)

    st.markdown('<br>', unsafe_allow_html=True)
    col_btn = st.columns([1, 2, 1])
    with col_btn[1]:
        if st.button("🎯  Start Exploring  →", use_container_width=True):
            st.session_state.page = "Discover"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 · DISCOVER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Discover":
    if not DATA_OK:
        st.error(f"Failed to load dataset: {DATA_ERR}")
        st.stop()

    st.markdown('<div class="hero-title" style="font-size:2.2rem">🔍 Discover Similar Songs</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Search for a song, select it, then generate lyric-based recommendations.</div>', unsafe_allow_html=True)

    lyric_col = next((c for c in df.columns if "lyric" in c.lower()), None)
    genre_col = detect_col(["genre"])
    year_col = detect_col(["year"])

    search_q = st.text_input(
        "",
        placeholder="Search by song name or artist…",
        label_visibility="collapsed",
        key="discover_search",
    )

    if getattr(rec, "backend", "cosine") == "hybrid":
        st.markdown(
            '<div style="font-size:0.75rem;color:#1DB954;margin-bottom:0.5rem;">● Hybrid model active</div>',
            unsafe_allow_html=True,
        )

    filtered = pd.DataFrame()
    if search_q and len(search_q) >= 1:
        try:
            filtered = rec.search(search_q)
        except Exception as e:
            st.error(f"Search error: {e}")

    selected_idx = None
    top_n = 10

    if not filtered.empty:
        st.markdown(f'<div class="section-sub">Found <b style="color:#1DB954">{len(filtered)}</b> results</div>', unsafe_allow_html=True)

        label_col = "song_name" if "song_name" in filtered.columns else filtered.columns[0]
        options = [
            f'{row[label_col]}  —  {strip_brackets(row.get("artist",""), 60)}'
            for _, row in filtered.iterrows()
        ]

        chosen = st.selectbox(
            "Select a song",
            options=options,
            label_visibility="visible",
        )
        chosen_idx_local = options.index(chosen)
        global_idx = filtered.index[chosen_idx_local]
        selected_idx = global_idx

        c1, c2 = st.columns([3, 1])
        with c2:
            top_n = st.select_slider("Results", options=[5, 10, 15, 20], value=10)

        if st.button("🎵  Generate Recommendations"):
            add_recent(
                filtered.iloc[chosen_idx_local].get("song_name", ""),
                filtered.iloc[chosen_idx_local].get("artist", ""),
            )
            st.session_state["last_recs_idx"] = int(global_idx)
            st.session_state["last_recs_n"] = top_n

    # show recs if available
    if "last_recs_idx" in st.session_state:
        try:
            ridx = st.session_state["last_recs_idx"]
            rn = st.session_state.get("last_recs_n", 10)
            results = rec.recommend(ridx, top_n=rn)

            seed_row = df.iloc[ridx]
            seed_lyric_raw = str(seed_row.get(lyric_col, "")).strip() if lyric_col else ""
            seed_lyric = html_lib.escape(seed_lyric_raw[:140]) if seed_lyric_raw not in ("nan", "None", "") else ""
            seed_meta_bits = []
            if genre_col:
                gv = strip_brackets(seed_row.get(genre_col, ""), 24)
                if gv:
                    seed_meta_bits.append(gv)
            if year_col:
                yv = str(seed_row.get(year_col, "")).strip()
                if yv not in ("", "nan", "None"):
                    try:
                        yv = str(int(float(yv)))
                    except ValueError:
                        pass
                    seed_meta_bits.append(yv)
            seed_meta = f'<div style="font-size:0.78rem;color:#1DB954;font-weight:600;margin-top:2px">{clean_field(" · ".join(seed_meta_bits), 40)}</div>' if seed_meta_bits else ""
            seed_lyric_html = f'<div style="font-size:0.78rem;color:#5a5a5a;margin-top:6px;line-height:1.4">{seed_lyric}</div>' if seed_lyric else ""

            st.markdown(f"""
            <div style="background:#181818;border:1px solid #1DB954;border-radius:14px;padding:1rem 1.25rem;margin:1rem 0;display:flex;align-items:flex-start;gap:12px;">
              <span style="font-size:1.8rem">🎵</span>
              <div>
                <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1rem;color:#fff">{seed_row.get('song_name','')}</div>
                <div style="font-size:0.82rem;color:#8a8a8a">Because you searched: <b style="color:#1DB954">{clean_field(seed_row.get('artist',''), 60)}</b></div>
                {seed_meta}
                {seed_lyric_html}
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f'<div class="section-header">Recommended for You · <span style="color:#1DB954">{len(results)} songs</span></div>', unsafe_allow_html=True)

            # merge lyric/genre/year cols if available (recommend() only returns song_name/artist/similarity)
            extra_cols = [c for c in (lyric_col, genre_col, year_col) if c and c in df.columns]
            if extra_cols:
                results = results.merge(
                    df[["song_name"] + extra_cols].rename(columns={"song_name": "_sn"}),
                    left_on="song_name", right_on="_sn", how="left"
                ).drop(columns=["_sn"], errors="ignore")

            render_song_grid(results.reset_index(drop=True), lyric_col=lyric_col, sim_col="similarity",
                              genre_col=genre_col, year_col=year_col, cols=4)

        except Exception as e:
            st.error(f"Recommendation error: {e}")

    elif not filtered.empty and search_q:
        pass  # waiting for button click
    elif not search_q:
        st.markdown("""
        <div class="no-results">
          <div class="icon">🎶</div>
          <div>Type a song name or artist to get started</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 · EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Explorer":
    if not DATA_OK:
        st.error(f"Failed to load dataset: {DATA_ERR}")
        st.stop()

    st.markdown('<div class="hero-title" style="font-size:2.2rem">🗂️ Song Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Browse the full dataset with search and filtering.</div>', unsafe_allow_html=True)

    lyric_col = next((c for c in df.columns if "lyric" in c.lower()), None)
    genre_col = detect_col(["genre"])
    year_col = detect_col(["year"])
    PAGE_SIZE = 24

    c1, c2 = st.columns([3, 1])
    with c1:
        ex_search = st.text_input("", placeholder="Filter by song or artist…", key="ex_search", label_visibility="collapsed")
    with c2:
        genres = ["All Genres"]
        if "genre" in df.columns:
            genres += sorted(df["genre"].dropna().unique().tolist())
        genre_filter = st.selectbox("Genre", genres, label_visibility="collapsed")

    # apply filters
    display_df = df[meta_cols].copy()
    if ex_search:
        q = ex_search.lower()
        mask = (
            display_df.get("song_name", pd.Series(dtype=str)).str.lower().str.contains(q, na=False) |
            display_df.get("artist", pd.Series(dtype=str)).str.lower().str.contains(q, na=False)
        )
        display_df = display_df[mask]
        st.session_state.explorer_page = 0
    if genre_filter != "All Genres" and "genre" in display_df.columns:
        display_df = display_df[display_df["genre"] == genre_filter]
        st.session_state.explorer_page = 0

    total = len(display_df)
    n_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    cur_page = min(st.session_state.explorer_page, n_pages - 1)

    st.markdown(f'<div class="section-sub"><b style="color:#1DB954">{total:,}</b> songs · Page {cur_page+1} of {n_pages}</div>', unsafe_allow_html=True)

    page_df = display_df.iloc[cur_page * PAGE_SIZE: (cur_page + 1) * PAGE_SIZE].reset_index(drop=True)
    render_song_grid(page_df, lyric_col=lyric_col, genre_col=genre_col, year_col=year_col)

    # pagination controls
    st.markdown("<br>", unsafe_allow_html=True)
    nav_cols = st.columns([2, 1, 1, 1, 2])
    with nav_cols[1]:
        if st.button("← Prev", disabled=(cur_page == 0), key="ex_prev"):
            st.session_state.explorer_page = max(0, cur_page - 1)
            st.rerun()
    with nav_cols[2]:
        st.markdown(f'<div style="text-align:center;padding:0.55rem;color:#8a8a8a;font-size:0.85rem">{cur_page+1} / {n_pages}</div>', unsafe_allow_html=True)
    with nav_cols[3]:
        if st.button("Next →", disabled=(cur_page >= n_pages - 1), key="ex_next"):
            st.session_state.explorer_page = min(n_pages - 1, cur_page + 1)
            st.rerun()

    # ── lyrics viewer ──
    if lyric_col:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📖 View Full Lyrics</div>', unsafe_allow_html=True)
        song_options = [
            f"{row['song_name']}  —  {strip_brackets(row.get('artist',''), 60)}"
            for _, row in page_df.iterrows()
            if str(row.get('song_name','')).strip() not in ('', 'nan')
        ]
        if song_options:
            picked = st.selectbox("Pick a song from this page", song_options, key="lyric_picker", label_visibility="collapsed")
            picked_idx = song_options.index(picked)
            full_lyric = str(page_df.iloc[picked_idx].get(lyric_col, ""))
            if full_lyric not in ("nan", "None", ""):
                with st.expander(f"🎵 {picked}", expanded=True):
                    st.markdown(
                        f'<div style="white-space:pre-wrap;color:#ccc;font-size:0.88rem;line-height:1.7;background:#181818;border-radius:10px;padding:1.5rem">{html_lib.escape(full_lyric)}</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.info("No lyrics available for this song.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 · ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Analytics":
    if not DATA_OK:
        st.error(f"Failed to load dataset: {DATA_ERR}")
        st.stop()

    st.markdown('<div class="hero-title" style="font-size:2.2rem">📊 Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Dataset insights and distribution analysis.</div>', unsafe_allow_html=True)

    DARK_BG = "#0f0f0f"
    CARD_BG = "#181818"
    GREEN   = "#1DB954"
    GREEN2  = "#4ade80"
    GRAY    = "#282828"
    TEXT    = "#ffffff"
    MUTED   = "#6a6a6a"

    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor":   CARD_BG,
        "axes.edgecolor":   GRAY,
        "axes.labelcolor":  MUTED,
        "xtick.color":      MUTED,
        "ytick.color":      MUTED,
        "text.color":       TEXT,
        "grid.color":       GRAY,
        "grid.linewidth":   0.5,
        "font.family":      "DejaVu Sans",
    })

    tab1, tab2, tab3 = st.tabs(["🎤 Top Artists", "🎭 Genre Distribution", "📐 Embedding Stats"])

    # ── top artists ──
    with tab1:
        if "artist" in df.columns:
            top_artists = df["artist"].apply(lambda v: strip_brackets(v, 60)).value_counts().head(20)
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = [GREEN if i < 3 else "#2a5c3a" for i in range(len(top_artists))]
            bars = ax.barh(top_artists.index[::-1], top_artists.values[::-1], color=colors[::-1], height=0.7)
            ax.set_xlabel("Number of Songs", fontsize=10, color=MUTED)
            ax.set_title("Top 20 Artists by Song Count", fontsize=14, fontweight="bold", color=TEXT, pad=15)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="x", alpha=0.3)
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.3, bar.get_y() + bar.get_height()/2, str(int(w)),
                        va="center", ha="left", fontsize=8, color=GREEN)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.warning("No 'artist' column found.")

    # ── genre ──
    with tab2:
        if "genre" in df.columns:
            genre_counts = df["genre"].value_counts().head(15)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

            # donut
            n = len(genre_counts)
            palette = plt.cm.Greens(np.linspace(0.3, 0.9, n))
            wedges, texts, autotexts = ax1.pie(
                genre_counts.values, labels=None,
                autopct="%1.1f%%", startangle=140,
                colors=palette,
                wedgeprops={"width": 0.65, "edgecolor": DARK_BG, "linewidth": 2},
                pctdistance=0.8,
            )
            for at in autotexts:
                at.set_fontsize(8); at.set_color(DARK_BG); at.set_fontweight("bold")
            ax1.set_title("Genre Share", fontsize=13, fontweight="bold", color=TEXT)
            ax1.legend(genre_counts.index, loc="lower left", fontsize=7,
                       facecolor=CARD_BG, edgecolor=GRAY, labelcolor=TEXT)

            # bar
            ax2.bar(range(len(genre_counts)), genre_counts.values,
                    color=[GREEN if i == 0 else "#2a5c3a" for i in range(len(genre_counts))],
                    width=0.7)
            ax2.set_xticks(range(len(genre_counts)))
            ax2.set_xticklabels(genre_counts.index, rotation=45, ha="right", fontsize=8)
            ax2.set_title("Songs per Genre", fontsize=13, fontweight="bold", color=TEXT)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.grid(axis="y", alpha=0.3)

            fig.patch.set_facecolor(DARK_BG)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("No 'genre' column found in dataset.")

    # ── embedding stats ──
    with tab3:
        try:
            sample_emb = rec.embeddings[:min(5000, len(rec.embeddings))]
            mean_vals = sample_emb.mean(axis=0)
            var_vals = sample_emb.var(axis=0)

            fig, axes = plt.subplots(1, 2, figsize=(13, 4))

            # top 20 highest-variance dims
            top_idx = np.argsort(-var_vals)[:20]
            axes[0].bar(range(20), var_vals[top_idx], color=GREEN, alpha=0.85, width=0.7)
            axes[0].set_xticks(range(20))
            axes[0].set_xticklabels([f"d{i}" for i in top_idx], rotation=45, ha="right", fontsize=7)
            axes[0].set_title("Top 20 Highest-Variance Dimensions", fontsize=12, fontweight="bold", color=TEXT)
            axes[0].set_xlabel("Dimension", color=MUTED)
            axes[0].set_ylabel("Variance", color=MUTED)
            axes[0].spines["top"].set_visible(False)
            axes[0].spines["right"].set_visible(False)
            axes[0].grid(axis="y", alpha=0.2)

            # mean embedding dims (first 80)
            x = np.arange(min(80, len(mean_vals)))
            axes[1].bar(x, mean_vals[:80], color=GREEN, alpha=0.7, width=1.0)
            axes[1].axhline(0, color=MUTED, linewidth=0.8)
            axes[1].set_title("Mean Embedding Values (first 80 dims)", fontsize=12, fontweight="bold", color=TEXT)
            axes[1].set_xlabel("Dimension", color=MUTED)
            axes[1].set_ylabel("Mean Value", color=MUTED)
            axes[1].spines["top"].set_visible(False)
            axes[1].spines["right"].set_visible(False)

            fig.patch.set_facecolor(DARK_BG)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.error(f"Embedding viz error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 · PROFILE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Profile":
    st.markdown("""
    <div class="profile-header">
      <div class="profile-avatar">🎧</div>
      <div>
        <div class="profile-name">Listener Profile</div>
        <div class="profile-sub">Your personalized listening history and stats</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    recent = st.session_state.get("recent_searches", [])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-value">{len(recent)}</div>
          <div class="stat-label">Songs Explored</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="stat-card">
          <div class="stat-value">∞</div>
          <div class="stat-label">Recommendations Available</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">🕐 Recent Searches</div>', unsafe_allow_html=True)

    if recent:
        st.markdown('<div class="section-sub">Last 10 songs you explored via Discover</div>', unsafe_allow_html=True)
        badges_html = "".join(
            f'<span class="hist-badge"><span class="dot">●</span>{entry}</span>'
            for entry in recent[:10]
        )
        st.markdown(f'<div style="margin:1rem 0">{badges_html}</div>', unsafe_allow_html=True)

        if DATA_OK and len(recent) >= 1:
            st.markdown('<div class="section-header">🎯 Based on Your History</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Songs similar to your recent searches</div>', unsafe_allow_html=True)
            try:
                # pick a random recent search and show recs
                last = recent[0]
                song_part = last.split(" — ")[0].strip()
                mask = df.get("song_name", pd.Series(dtype=str)).str.lower() == song_part.lower()
                hits = df[mask]
                if not hits.empty:
                    ridx = hits.index[0]
                    suggestions = rec.recommend(ridx, top_n=8)
                    lyric_col = next((c for c in df.columns if "lyric" in c.lower()), None)
                    genre_col = detect_col(["genre"])
                    year_col = detect_col(["year"])
                    extra_cols = [c for c in (lyric_col, genre_col, year_col) if c and c in df.columns]
                    if extra_cols:
                        suggestions = suggestions.merge(
                            df[["song_name"] + extra_cols].rename(columns={"song_name": "_sn"}),
                            left_on="song_name", right_on="_sn", how="left"
                        ).drop(columns=["_sn"], errors="ignore")
                    render_song_grid(suggestions.reset_index(drop=True), lyric_col=lyric_col, sim_col="similarity",
                                      genre_col=genre_col, year_col=year_col, cols=4)
                else:
                    st.info("No exact match found for most recent search.")
            except Exception as e:
                st.error(f"Profile recs error: {e}")
    else:
        st.markdown("""
        <div class="no-results">
          <div class="icon">🎵</div>
          <div>No history yet — head to <b>Discover</b> and explore some songs!</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️  Clear History"):
        st.session_state.recent_searches = []
        st.rerun()
