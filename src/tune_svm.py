import pandas as pd
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ========================================
# 1. Load data
# ========================================

X_train = pd.read_csv("data/processed/steam_train.csv")
X_test = pd.read_csv("data/processed/steam_test.csv")

y_train = X_train.pop("success")
y_test = X_test.pop("success")


# ========================================
# 2. Load preprocessing pipeline
# ========================================

preprocessor = joblib.load(
    "models/steam_preprocessor.joblib"
)


# ========================================
# 3. Transform data
# ========================================

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# ========================================
# 4. Define SVM
# ========================================

svm = SVC(
    kernel="rbf",
    class_weight="balanced",
    random_state=42
)


# ========================================
# 5. Hyperparameter grid
# ========================================

param_grid = {
    "C": [0.1, 1, 5, 10, 20],
    "gamma": ["scale", 0.01, 0.1]
}


# ========================================
# 6. Grid Search
# ========================================

grid_search = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    verbose=1
)


grid_search.fit(
    X_train_processed,
    y_train
)


# ========================================
# 7. Best parameters
# ========================================

print("\n========================================")
print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(grid_search.best_score_)


# ========================================
# 8. Test set evaluation
# ========================================

best_svm = grid_search.best_estimator_

y_pred = best_svm.predict(X_test_processed)

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n========================================")
print("Test Results")
print("========================================")

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
print(confusion_matrix(y_test, y_pred))


# ========================================
# 9. Save model
# ========================================

joblib.dump(
    best_svm,
    "models/steam_svm_tuned.joblib"
)

print("\nModel saved as:")
print("steam_svm_tuned.joblib")