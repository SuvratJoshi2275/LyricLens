"""Generate a safe synthetic song dataset for the college demo.

The rows are original lyric-style snippets, not copied from real songs.
This keeps the project distributable while still giving Sentence-BERT
enough semantic signal to demonstrate cross-genre recommendation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/songs.csv")


GENRES = ["metal", "rock", "hip-hop", "pop", "hindi", "punjabi", "indie"]


THEME_BANK = {
    "heartbreak": {
        "nouns": ["goodbye", "breakup", "broken heart", "closed door", "old photograph"],
        "lines": [
            "you left me standing where the rain wrote goodbye on the glass",
            "the broken heart keeps knocking on a door that will not open",
            "every promise sounds distant after the breakup burned the room",
        ],
    },
    "loneliness": {
        "nouns": ["empty room", "silence", "shadow", "hollow street", "no one"],
        "lines": [
            "i walk alone through a hollow street while the city forgets my name",
            "the empty room answers back with silence and no one calls",
            "a shadow sits beside me like a friend who never speaks",
        ],
    },
    "loss/grief": {
        "nouns": ["ashes", "grave", "memory", "gone voice", "mourning"],
        "lines": [
            "your picture stays warm while the house learns the shape of grief",
            "i miss you in the morning when the gone voice fills the air",
            "ashes of memory drift over the road beside the grave",
        ],
    },
    "guilt/regret": {
        "nouns": ["sorry", "fault", "regret", "blame", "apology"],
        "lines": [
            "i stare at the message i never sent and wish forgiveness could answer",
            "regret becomes a map of the mistake i keep trying to understand",
            "my apology circles the ceiling while shame writes my name",
        ],
    },
    "self-reflection": {
        "nouns": ["mirror", "truth", "inside", "question", "soul"],
        "lines": [
            "the mirror asks a question that i have avoided for years",
            "inside my quiet soul i search for truth and learn to change",
            "every reflection teaches me the person i was and the person i can be",
        ],
    },
    "ambition": {
        "nouns": ["dream", "city", "crown", "goal", "future"],
        "lines": [
            "i chase the dream through the city with a goal under my tongue",
            "the future shines like a crown above the stage lights",
            "every early morning grind builds a legacy from dust",
        ],
    },
    "struggle": {
        "nouns": ["battle", "pressure", "scar", "hard road", "survive"],
        "lines": [
            "the hard road puts pressure on my chest but i survive",
            "every scar becomes a map of the battle i kept walking through",
            "i fight the heavy night and carry pain until it turns to strength",
        ],
    },
}


EMOTION_BANK = {
    "happy": [
        "a bright smile breaks open the day and the whole room starts to dance",
        "joy runs through the chorus like sunshine after a long sleep",
        "i feel free enough to celebrate every small light",
    ],
    "sad": [
        "tears move slowly because the heart feels empty and blue",
        "the hollow night keeps the pain close and the voice begins to cry",
        "sad memories gather softly around the bed",
    ],
    "angry": [
        "rage burns like fire and the storm inside starts to shout",
        "i fight the hate before revenge can turn me into war",
        "angry sparks hit the wall while the drums push harder",
    ],
    "calm": [
        "i breathe slowly beside a quiet river and let peace return",
        "the moon feels gentle and the room becomes still",
        "soft light moves through the calm evening without a fight",
    ],
    "motivational": [
        "i rise with a strong belief and climb toward the goal",
        "the hustle teaches me to build a dream one step at a time",
        "a champion voice says win again even after the fall",
    ],
    "romantic": [
        "love holds the heart close and calls my darling home",
        "your kiss feels like forever when we stand together",
        "beloved hands turn the night into a gentle romance",
    ],
}


GENRE_VOCAB = {
    "metal": ["distorted", "thunder", "scream", "iron", "blast"],
    "rock": ["guitar", "highway", "amplifier", "neon", "chorus"],
    "hip-hop": ["beat", "verse", "block", "mic", "bass"],
    "pop": ["radio", "hook", "dance floor", "glitter", "chorus"],
    "hindi": ["sheher", "raat", "dil", "yaad", "safar"],
    "punjabi": ["dhol", "yaar", "pind", "akhiyan", "raah"],
    "indie": ["bedroom", "cassette", "window", "coffee", "paper moon"],
}


TITLE_PARTS = {
    "heartbreak": ["Door", "Farewell", "Broken", "After You", "Rain"],
    "loneliness": ["Hollow", "Alone", "Empty", "Shadow", "Silent"],
    "loss/grief": ["Ashes", "Gone", "Memory", "Mourning", "Grave"],
    "guilt/regret": ["Apology", "Fault", "Regret", "Blame", "Sorry"],
    "self-reflection": ["Mirror", "Question", "Inside", "Truth", "Change"],
    "ambition": ["Crown", "Dream", "Future", "Legacy", "City"],
    "struggle": ["Battle", "Scar", "Pressure", "Survive", "Road"],
}


ARTIST_PREFIX = {
    "metal": ["Iron Veil", "Ashen Circuit", "Black Voltage", "Rusted Crown"],
    "rock": ["Neon Harbor", "Static Avenue", "The Late Signals", "Crimson Radio"],
    "hip-hop": ["Metro Verse", "Northside Echo", "Cipher Lane", "Maya Bars"],
    "pop": ["Luna Vale", "Skyline Hearts", "Nova June", "Mira Bloom"],
    "hindi": ["Aarav Saanjh", "Meera Raahi", "Kabir Dhun", "Nisha Safar"],
    "punjabi": ["Diljit Raah", "Amrit Noor", "Gagan Pind", "Simran Akhiyan"],
    "indie": ["Paper Lanterns", "Bedroom Atlas", "Juniper Days", "The Soft Frames"],
}


DEMO_ROWS = [
    {
        "song_name": "Empty Furnace",
        "artist": "Ashen Circuit",
        "genre": "metal",
        "lyrics": (
            "Distorted thunder shakes the room while I carry blame in my pocket. "
            "I whisper sorry to the dark because the mistake was my fault. "
            "Regret keeps replaying in a hollow chest, empty and blue, "
            "and every scream asks for forgive me after the fire is gone."
        ),
    },
    {
        "song_name": "Backseat Apology",
        "artist": "Metro Verse",
        "genre": "hip-hop",
        "lyrics": (
            "The bass hits like distorted thunder while I carry blame in my pocket. "
            "I whisper sorry to the dark because the mistake was my fault. "
            "Regret keeps replaying in a hollow chest, empty and blue, "
            "and every verse asks for forgive me after the fire is gone."
        ),
    },
    {
        "song_name": "Window Without Voices",
        "artist": "Bedroom Atlas",
        "genre": "indie",
        "lyrics": (
            "Distorted silence shakes the empty room while I carry a hollow night in my pocket. "
            "I whisper to the dark because no one answers from the window. "
            "The same memory keeps replaying in a hollow chest, empty and blue, "
            "and every quiet shadow asks why no one came home after the fire was gone."
        ),
    },
    {
        "song_name": "Akhiyan Di Baarish",
        "artist": "Simran Akhiyan",
        "genre": "punjabi",
        "lyrics": (
            "Distorted thunder shakes the room while the dhol waits on the raah back to the pind. "
            "I carry blame in my pocket and whisper sorry to the dark because the mistake was my fault. "
            "Regret keeps replaying in a hollow chest, empty and blue, "
            "and every yaad asks for forgive me after the fire is gone."
        ),
    },
    {
        "song_name": "Sheher Ki Yaad",
        "artist": "Meera Raahi",
        "genre": "hindi",
        "lyrics": (
            "In the sheher at raat I hold one yaad and miss you. "
            "The dil feels grief because your gone voice fills the safar. "
            "I cry near the old station where memory turns to ashes, "
            "and the sad road keeps your name alive."
        ),
    },
    {
        "song_name": "Neon Goodbye",
        "artist": "Luna Vale",
        "genre": "pop",
        "lyrics": (
            "The radio hook keeps saying goodbye under glitter lights. "
            "You left me with a broken heart on the dance floor. "
            "I smile for the camera but the breakup feels blue, "
            "and without you every chorus falls apart."
        ),
    },
]


def build_lyrics(theme: str, emotion: str, genre: str, variant: int) -> str:
    theme_data = THEME_BANK[theme]
    emotion_lines = EMOTION_BANK[emotion]
    genre_words = GENRE_VOCAB[genre]

    theme_line_a = theme_data["lines"][variant % len(theme_data["lines"])]
    theme_line_b = theme_data["lines"][(variant + 1) % len(theme_data["lines"])]
    emotion_line = emotion_lines[variant % len(emotion_lines)]
    vocab = genre_words[variant % len(genre_words)]
    vocab_next = genre_words[(variant + 2) % len(genre_words)]
    noun = theme_data["nouns"][variant % len(theme_data["nouns"])]

    return (
        f"{theme_line_a}. {emotion_line}. "
        f"The {genre} texture carries {vocab} and {vocab_next} around the word {noun}. "
        f"{theme_line_b}. I return to the same feeling again because the meaning matters more than the sound."
    )


def build_title(theme: str, emotion: str, genre: str, variant: int) -> str:
    theme_word = TITLE_PARTS[theme][variant % len(TITLE_PARTS[theme])]
    genre_tag = genre.title().replace("-", " ")
    suffixes = {
        "happy": "Light",
        "sad": "Blue",
        "angry": "Fire",
        "calm": "Moon",
        "motivational": "Rise",
        "romantic": "Heart",
    }
    if variant % 3 == 0:
        return f"{theme_word} {suffixes[emotion]} {genre_tag}"
    if variant % 3 == 1:
        return f"{genre_tag} {theme_word}"
    return f"{theme_word} {variant + 1}"


def create_sample_dataset(output_path: Path = DATA_PATH) -> pd.DataFrame:
    rows = list(DEMO_ROWS)
    seen = {(row["song_name"], row["artist"]) for row in rows}

    for theme in THEME_BANK:
        for emotion in EMOTION_BANK:
            for genre in GENRES:
                for variant in range(2):
                    artist = ARTIST_PREFIX[genre][(variant + len(theme) + len(emotion)) % len(ARTIST_PREFIX[genre])]
                    title = build_title(theme, emotion, genre, variant)
                    if (title, artist) in seen:
                        title = f"{title} {len(rows)}"
                    seen.add((title, artist))
                    rows.append(
                        {
                            "song_name": title,
                            "artist": artist,
                            "genre": genre,
                            "lyrics": build_lyrics(theme, emotion, genre, variant),
                        }
                    )

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    dataset = create_sample_dataset()
    print(f"Created {len(dataset)} songs at {DATA_PATH}")
