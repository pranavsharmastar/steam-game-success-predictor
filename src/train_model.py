import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------------
# 1. Load the training and test data
# -----------------------------------

train = pd.read_csv("data/processed/steam_train.csv")
test = pd.read_csv("data/processed/steam_test.csv")

# Separate features and target
X_train = train.drop("success", axis=1)
y_train = train["success"]

X_test = test.drop("success", axis=1)
y_test = test["success"]


# -----------------------------------
# 2. Load our preprocessing pipeline
# -----------------------------------

preprocessor = joblib.load("models/steam_preprocessor.joblib")


# -----------------------------------
# 3. Preprocess the data
# -----------------------------------

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# -----------------------------------
# 4. Create Logistic Regression model
# -----------------------------------

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)


# -----------------------------------
# 5. Train the model
# -----------------------------------

model.fit(X_train_processed, y_train)


# -----------------------------------
# 6. Make predictions
# -----------------------------------

y_pred = model.predict(X_test_processed)


# -----------------------------------
# 7. Evaluate the model
# -----------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("Logistic Regression Results")
print("=" * 40)

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not Successful", "Successful"]
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# -----------------------------------
# 8. Save the trained model
# -----------------------------------

joblib.dump(model, "steam_logistic_regression.joblib")

print("\nModel saved as:")
print("steam_logistic_regression.joblib")