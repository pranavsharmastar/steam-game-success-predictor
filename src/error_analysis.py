import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix


# ========================================
# 1. Load test data
# ========================================

test_df = pd.read_csv(
    "data/processed/steam_test.csv"
)


# Separate target

y_test = test_df.pop("success")


# ========================================
# 2. Load preprocessing pipeline
# ========================================

preprocessor = joblib.load(
    "models/steam_preprocessor.joblib"
)


# ========================================
# 3. Load trained Logistic Regression
# ========================================

model = joblib.load(
    "models/steam_logistic_regression_tuned.joblib"
)


# ========================================
# 4. Transform test data
# ========================================

X_test_processed = preprocessor.transform(
    test_df
)


# ========================================
# 5. Make predictions
# ========================================

y_pred = model.predict(
    X_test_processed
)


# ========================================
# 6. Add predictions
# ========================================

results = test_df.copy()

results["actual"] = y_test.values
results["predicted"] = y_pred


# ========================================
# 7. Identify error types
# ========================================

false_positives = results[
    (results["actual"] == 0) &
    (results["predicted"] == 1)
]

false_negatives = results[
    (results["actual"] == 1) &
    (results["predicted"] == 0)
]

correct_predictions = results[
    results["actual"] == results["predicted"]
]


# ========================================
# 8. Print basic error counts
# ========================================

print("========================================")
print("ERROR ANALYSIS")
print("========================================")

print(
    "\nTotal test samples:",
    len(results)
)

print(
    "Correct predictions:",
    len(correct_predictions)
)

print(
    "False positives:",
    len(false_positives)
)

print(
    "False negatives:",
    len(false_negatives)
)


# ========================================
# 9. Confusion matrix
# ========================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ========================================
# 10. Error percentages
# ========================================

print("\n========================================")
print("ERROR PERCENTAGES")
print("========================================")

print(
    "False positive rate:",
    round(
        len(false_positives) /
        len(results) * 100,
        2
    ),
    "%"
)

print(
    "False negative rate:",
    round(
        len(false_negatives) /
        len(results) * 100,
        2
    ),
    "%"
)


# ========================================
# 11. Feature-level comparison
# ========================================

numeric_columns = [
    "release_year",
    "achievements",
    "original_price",
    "is_free"
]


print("\n========================================")
print("NUMERIC FEATURE COMPARISON")
print("========================================")


for column in numeric_columns:

    print("\n", column)

    print(
        "Actual Successful:",
        round(
            results.loc[
                results["actual"] == 1,
                column
            ].mean(),
            2
        )
    )

    print(
        "Actual Not Successful:",
        round(
            results.loc[
                results["actual"] == 0,
                column
            ].mean(),
            2
        )
    )


# ========================================
# 12. False Positive feature comparison
# ========================================

print("\n========================================")
print("FALSE POSITIVE ANALYSIS")
print("========================================")


for column in numeric_columns:

    print("\n", column)

    print(
        "False Positive:",
        round(
            false_positives[column].mean(),
            2
        )
    )

    print(
        "True Negative:",
        round(
            results.loc[
                (results["actual"] == 0) &
                (results["predicted"] == 0),
                column
            ].mean(),
            2
        )
    )


# ========================================
# 13. False Negative feature comparison
# ========================================

print("\n========================================")
print("FALSE NEGATIVE ANALYSIS")
print("========================================")


for column in numeric_columns:

    print("\n", column)

    print(
        "False Negative:",
        round(
            false_negatives[column].mean(),
            2
        )
    )

    print(
        "True Positive:",
        round(
            results.loc[
                (results["actual"] == 1) &
                (results["predicted"] == 1),
                column
            ].mean(),
            2
        )
    )


# ========================================
# 14. Save complete predictions
# ========================================

results.to_csv(
    "data/processed/steam_prediction_errors.csv",
    index=False
)


print("\n========================================")
print("Saved:")
print("data/processed/steam_prediction_errors.csv")
print("========================================")