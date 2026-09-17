import pandas as pd
import numpy as np
import joblib


# ============================================================
# 1. File paths
# ============================================================

data_file = "data/processed/steam_train.csv"

preprocessor_file = (
    "models/steam_preprocessor.joblib"
)

model_file = (
    "models/steam_logistic_regression.joblib"
)


# ============================================================
# 2. Load data, preprocessor and model
# ============================================================

print("Loading data and model...")

df = pd.read_csv(data_file)

preprocessor = joblib.load(
    preprocessor_file
)

model = joblib.load(
    model_file
)


# ============================================================
# 3. Separate features and target
# ============================================================

X = df.drop(
    columns=["success"]
)

y = df["success"]


# ============================================================
# 4. Transform features
# ============================================================

X_transformed = preprocessor.transform(X)


# ============================================================
# 5. Get feature names
# ============================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)


# ============================================================
# 6. Get Logistic Regression coefficients
# ============================================================

coefficients = model.coef_[0]


# ============================================================
# 7. Check dimensions
# ============================================================

print("\n")
print("=" * 60)
print("DIMENSION CHECK")
print("=" * 60)

print(
    "Transformed feature count:",
    X_transformed.shape[1]
)

print(
    "Feature name count:",
    len(feature_names)
)

print(
    "Model coefficient count:",
    len(coefficients)
)


# ============================================================
# 8. Make sure dimensions match
# ============================================================

if len(feature_names) != len(coefficients):

    print("\nERROR:")
    print(
        "The model and preprocessor have different "
        "numbers of features."
    )

    print(
        "\nThis usually means the model was trained "
        "with an older version of the preprocessing."
    )

    print(
        "\nPlease retrain the model before running "
        "feature analysis."
    )

    raise SystemExit


# ============================================================
# 9. Create feature importance dataframe
# ============================================================

importance_df = pd.DataFrame({

    "feature": feature_names,

    "coefficient": coefficients,

    "absolute_importance": np.abs(
        coefficients
    )

})


# ============================================================
# 10. Sort by importance
# ============================================================

importance_df = (
    importance_df
    .sort_values(
        by="absolute_importance",
        ascending=False
    )
)


# ============================================================
# 11. Top 20 features overall
# ============================================================

print("\n")
print("=" * 60)
print("TOP 20 FEATURES BY IMPORTANCE")
print("=" * 60)

print(
    importance_df[
        [
            "feature",
            "coefficient"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 12. Features supporting success
# ============================================================

positive_features = (
    importance_df[
        importance_df["coefficient"] > 0
    ]
    .sort_values(
        by="coefficient",
        ascending=False
    )
)


print("\n")
print("=" * 60)
print("TOP 15 FEATURES SUPPORTING SUCCESS")
print("=" * 60)

print(
    positive_features[
        [
            "feature",
            "coefficient"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 13. Features against success
# ============================================================

negative_features = (
    importance_df[
        importance_df["coefficient"] < 0
    ]
    .sort_values(
        by="coefficient",
        ascending=True
    )
)


print("\n")
print("=" * 60)
print("TOP 15 FEATURES AGAINST SUCCESS")
print("=" * 60)

print(
    negative_features[
        [
            "feature",
            "coefficient"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 14. Group feature importance
# ============================================================

def get_feature_group(feature):

    if feature.startswith("num__"):
        return "Numerical"

    elif feature.startswith("developer__"):
        return "Developer"

    elif feature.startswith("publisher__"):
        return "Publisher"

    elif feature.startswith("cat__"):
        return "Categorical"

    elif feature.startswith("binary__genre_"):
        return "Genre"

    elif feature.startswith("binary__tag_"):
        return "Tag"

    else:
        return "Other"


importance_df["feature_group"] = (
    importance_df["feature"]
    .apply(get_feature_group)
)


# ============================================================
# 15. Average importance by group
# ============================================================

group_importance = (
    importance_df
    .groupby("feature_group")[
        "absolute_importance"
    ]
    .mean()
    .sort_values(
        ascending=False
    )
)


print("\n")
print("=" * 60)
print("AVERAGE IMPORTANCE BY FEATURE GROUP")
print("=" * 60)

print(
    group_importance.to_string()
)


# ============================================================
# 16. Save results
# ============================================================

output_file = (
    "data/processed/feature_importance.csv"
)

importance_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 17. Finished
# ============================================================

print("\n")
print("=" * 60)
print("FEATURE ANALYSIS COMPLETE")
print("=" * 60)

print(
    "\nFull feature importance saved to:"
)

print(output_file)

