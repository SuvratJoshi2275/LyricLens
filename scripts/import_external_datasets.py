"""Merge downloaded lyric datasets into the app's required CSV format.

Usage:
    python import_external_datasets.py raw_data/dataset1.csv raw_data/dataset2.csv

Output:
    data/songs.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = ["song_name", "artist", "lyrics", "genre"]

COLUMN_ALIASES = {
    "song_name": ["song_name", "song", "track_name", "track", "title", "name"],
    "artist": ["artist", "artist_name", "singer", "artists", "band"],
    "lyrics": ["lyrics", "lyric", "text", "clean_lyrics", "song_lyrics"],
    "genre": ["genre", "genres", "music_genre", "tag", "class", "category"],
}

TARGET_GENRES = {
    "rock": ["rock", "alternative rock", "classic rock", "hard rock"],
    "metal": ["metal", "heavy metal", "death metal", "black metal", "nu metal"],
    "hip-hop": ["hip hop", "hip-hop", "rap"],
    "pop": ["pop", "dance pop", "electropop", "synthpop"],
    "country": ["country"],
}


def find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    normalized = {column.lower().strip(): column for column in df.columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def normalize_genre(value: object) -> str:
    text = str(value).lower().strip()
    for target, aliases in TARGET_GENRES.items():
        if any(alias in text for alias in aliases):
            return target
    return text.split(",")[0].strip() if text else "unknown"


def load_one_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    mapping = {}
    for output_column, aliases in COLUMN_ALIASES.items():
        source = find_column(df, aliases)
        if source is None:
            raise ValueError(
                f"{path} is missing a usable '{output_column}' column. "
                f"Available columns: {list(df.columns)}"
            )
        mapping[source] = output_column

    cleaned = df.rename(columns=mapping)[OUTPUT_COLUMNS].copy()
    cleaned["genre"] = cleaned["genre"].apply(normalize_genre)
    return cleaned


def merge_datasets(paths: list[Path], sample_per_genre: int, output_path: Path) -> pd.DataFrame:
    frames = [load_one_csv(path) for path in paths]
    merged = pd.concat(frames, ignore_index=True)

    merged = merged.dropna(subset=OUTPUT_COLUMNS)
    for column in OUTPUT_COLUMNS:
        merged[column] = merged[column].astype(str).str.strip()

    merged = merged[
        (merged["song_name"] != "")
        & (merged["artist"] != "")
        & (merged["lyrics"].str.split().str.len() >= 20)
        & (merged["genre"] != "")
        & (merged["genre"] != "unknown")
    ].copy()

    merged = merged.drop_duplicates(subset=["song_name", "artist"], keep="first")
    merged = (
        merged.groupby("genre", group_keys=False)
        .apply(lambda group: group.sample(min(len(group), sample_per_genre), random_state=42))
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files", nargs="+", type=Path)
    parser.add_argument("--sample-per-genre", type=int, default=120)
    parser.add_argument("--output", type=Path, default=Path("data/songs.csv"))
    args = parser.parse_args()

    merged = merge_datasets(args.csv_files, args.sample_per_genre, args.output)
    print(f"Saved {len(merged)} songs to {args.output}")
    print(merged["genre"].value_counts().to_string())


if __name__ == "__main__":
    main()
