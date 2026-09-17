import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# ========================================
# 1. Load final dataset
# ========================================

df = pd.read_csv(
    "data/processed/steam_game_success_final.csv"
)


# ========================================
# 2. Separate features and target
# ========================================

X = df.drop(columns=["success"])
y = df["success"]


# ========================================
# 3. Feature groups
# ========================================

numeric_features = [
    "release_year",
    "achievements",
    "achievements_missing",
    "original_price",
    "is_free"
]


genre_features = [
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


tag_features = [
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


categorical_features = [
    "developer",
    "publisher",
    "release_era"
]


# ========================================
# 4. Numeric preprocessing
# ========================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ========================================
# 5. Binary feature preprocessing
# ========================================

binary_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        )
    ]
)


# ========================================
# 6. Developer / Publisher preprocessing
# ========================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Unknown"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=5
            )
        )
    ]
)


# ========================================
# 7. Complete preprocessing pipeline
# ========================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),

        (
            "genres",
            binary_pipeline,
            genre_features
        ),

        (
            "tags",
            binary_pipeline,
            tag_features
        ),

        (
            "developer_publisher",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ========================================
# 8. Train / Test split
# ========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# ========================================
# 9. Save RAW train/test datasets
# ========================================

train_df = X_train.copy()
train_df["success"] = y_train.values

test_df = X_test.copy()
test_df["success"] = y_test.values


train_df.to_csv(
    "data/processed/steam_train.csv",
    index=False
)

test_df.to_csv(
    "data/processed/steam_test.csv",
    index=False
)


# ========================================
# 10. Fit preprocessor on training data
# ========================================

preprocessor.fit(X_train)


# ========================================
# 11. Save preprocessor
# ========================================

joblib.dump(
    preprocessor,
    "models/steam_preprocessor.joblib"
)


# ========================================
# 12. Display information
# ========================================

print("========================================")
print("Preprocessing completed")
print("========================================")

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)

print(
    "Training features:",
    X_train.shape[1]
)

print(
    "Testing features:",
    X_test.shape[1]
)

print("\nTrain dataset saved:")
print("data/processed/steam_train.csv")

print("\nTest dataset saved:")
print("data/processed/steam_test.csv")

print("\nPreprocessor saved:")
print("models/steam_preprocessor.joblib")