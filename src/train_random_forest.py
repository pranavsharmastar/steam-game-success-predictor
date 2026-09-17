import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. Load training and testing data
# ============================================================

train_file = "data/processed/steam_train.csv"
test_file = "data/processed/steam_test.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)


# ============================================================
# 2. Separate features and target
# ============================================================

X_train = train_df.drop("success", axis=1)
y_train = train_df["success"]

X_test = test_df.drop("success", axis=1)
y_test = test_df["success"]


# ============================================================
# 3. Load preprocessing pipeline
# ============================================================

preprocessor = joblib.load(
    "models/steam_preprocessor.joblib"
)


# ============================================================
# 4. Transform the data
# ============================================================

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# ============================================================
# 5. Create Random Forest model
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


# ============================================================
# 6. Train model
# ============================================================

model.fit(
    X_train_processed,
    y_train
)


# ============================================================
# 7. Make predictions
# ============================================================

y_pred = model.predict(X_test_processed)


# ============================================================
# 8. Evaluate model
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nRandom Forest Results")
print("=" * 40)

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Not Successful",
            "Successful"
        ]
    )
)


print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 9. Save model
# ============================================================

joblib.dump(
    model,
    "models/steam_random_forest.joblib"
)

print("\nModel saved as:")
print("steam_random_forest.joblib")

