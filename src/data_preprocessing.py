import pandas as pd
import numpy as np
import re


# ============================================================
# 1. Load raw dataset
# ============================================================

input_file = "data/raw/steam_games.csv"
output_file = "data/processed/steam_game_success_final.csv"

df = pd.read_csv(input_file)

print("Original dataset shape:", df.shape)


# ============================================================
# 2. Keep only actual Steam apps
# ============================================================

df = df[df["types"] == "app"].copy()

print("After keeping apps:", df.shape)


# ============================================================
# 3. Extract review count
# ============================================================

def extract_review_count(text):

    if pd.isna(text):
        return np.nan

    match = re.search(r"\(([\d,]+)\)", str(text))

    if match:
        return int(match.group(1).replace(",", ""))

    return np.nan


df["review_count"] = df["all_reviews"].apply(
    extract_review_count
)


# ============================================================
# 4. Extract release year
# ============================================================

def extract_year(value):

    if pd.isna(value):
        return np.nan

    match = re.search(r"(19|20)\d{2}", str(value))

    if match:
        return int(match.group())

    return np.nan


df["release_year"] = df["release_date"].apply(
    extract_year
)


# Keep years with enough usable data

df = df[
    (df["release_year"] >= 2006) &
    (df["release_year"] <= 2019)
].copy()


# ============================================================
# 4.1 Create release era
# ============================================================

def get_release_era(year):

    if 2006 <= year <= 2010:
        return "2006-2010"

    elif 2011 <= year <= 2013:
        return "2011-2013"

    elif 2014 <= year <= 2016:
        return "2014-2016"

    elif 2017 <= year <= 2019:
        return "2017-2019"

    else:
        return "Unknown"


df["release_era"] = df["release_year"].apply(
    get_release_era
)


# ============================================================
# 5. Create success target
# ============================================================

# Calculate threshold using only games
# with known review counts

SUCCESS_THRESHOLD = (
    df["review_count"]
    .dropna()
    .quantile(0.75)
)

print("Success threshold:", SUCCESS_THRESHOLD)


# Only games with known review count
# can be classified

df = df[df["review_count"].notna()].copy()


df["success"] = (
    df["review_count"] >= SUCCESS_THRESHOLD
).astype(int)


# ============================================================
# 6. Clean original price
# ============================================================

def extract_price(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    # Free games
    if value.lower() == "free":
        return 0.0

    # Extract a valid number
    match = re.search(
        r"\d+(?:\.\d+)?",
        value
    )

    if match:

        try:

            price = float(match.group())

            # Remove unrealistic/malformed values
            if price <= 200:
                return price

        except ValueError:
            pass

    return np.nan


df["original_price"] = df[
    "original_price"
].apply(extract_price)


# Free game indicator

df["is_free"] = (
    df["original_price"] == 0
).astype(int)


# ============================================================
# 7. Clean achievements
# ============================================================

df["achievements"] = pd.to_numeric(
    df["achievements"],
    errors="coerce"
)


# Keep track of missing achievements

df["achievements_missing"] = (
    df["achievements"]
    .isna()
    .astype(int)
)


# ============================================================
# 8. Convert GENRE combinations
#    into individual features
# ============================================================

genres = [

    "Action",
    "Adventure",
    "Casual",
    "Indie",
    "RPG",
    "Simulation",
    "Strategy",
    "Sports",
    "Racing",
    "Free to Play",
    "Massively Multiplayer",
    "Early Access",
    "Utilities"

]


for genre in genres:

    column_name = (
        "genre_" +
        genre.lower().replace(" ", "_")
    )

    df[column_name] = (
        df["genre"]
        .fillna("")
        .str.contains(
            re.escape(genre),
            case=False,
            regex=True
        )
        .astype(int)
    )


# ============================================================
# 9. Convert popular tags
#    into individual features
# ============================================================

tags = [

    "Action",
    "Indie",
    "Adventure",
    "RPG",
    "Strategy",
    "Casual",
    "Simulation",
    "Sports",
    "Racing",
    "Multiplayer",
    "FPS",
    "Shooter",
    "Singleplayer"

]


for tag in tags:

    column_name = (
        "tag_" +
        tag.lower()
    )

    df[column_name] = (
        df["popular_tags"]
        .fillna("")
        .str.contains(
            re.escape(tag),
            case=False,
            regex=True
        )
        .astype(int)
    )


# ============================================================
# 10. Select final features
# ============================================================

feature_columns = [

    # Basic numerical features

    "release_year",

    # New feature:
    # broader release period

    "release_era",

    "achievements",

    "achievements_missing",

    "original_price",

    "is_free",


    # Developer and publisher

    "developer",

    "publisher"

]


# Add genre features

feature_columns += [

    "genre_action",
    "genre_adventure",
    "genre_casual",
    "genre_indie",
    "genre_rpg",
    "genre_simulation",
    "genre_strategy",
    "genre_sports",
    "genre_racing",
    "genre_free_to_play",
    "genre_massively_multiplayer",
    "genre_early_access",
    "genre_utilities"

]


# Add tag features

feature_columns += [

    "tag_action",
    "tag_indie",
    "tag_adventure",
    "tag_rpg",
    "tag_strategy",
    "tag_casual",
    "tag_simulation",
    "tag_sports",
    "tag_racing",
    "tag_multiplayer",
    "tag_fps",
    "tag_shooter",
    "tag_singleplayer"

]


# Target

feature_columns.append("success")


# Create final dataset

final_df = df[feature_columns].copy()


# ============================================================
# 11. Save final dataset
# ============================================================

final_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 12. Display results
# ============================================================

print(
    "\nFinal dataset shape:",
    final_df.shape
)


print("\nSuccess distribution:")

print(
    final_df["success"].value_counts()
)


print("\nSuccess percentage:")

print(
    final_df["success"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


print("\nRelease era distribution:")

print(
    final_df["release_era"].value_counts()
)


print("\nFinal features:")

print(
    final_df.columns.tolist()
)


print("\nSaved to:")

print(output_file)

