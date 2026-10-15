import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATA
# ============================================================

train_data = pd.read_parquet(
    "data/train-00000-of-00001.parquet"
)

test_data = pd.read_parquet(
    "data/test-00000-of-00001.parquet"
)

print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

features = [
    "Year",
    "State",
    "Crop",
    "Season",
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]

target = "Yield"

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

print("\nX_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("\nFeatures:")
print(features)

print("\nTarget:")
print(target)


# ============================================================
# 3. PREPROCESSING
# ============================================================

categorical_features = [
    "State",
    "Crop",
    "Season"
]

numerical_features = [
    "Year",
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

print("\nPreprocessing created successfully!")


# ============================================================
# 4. RANDOM FOREST MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# 5. TRAIN NORMAL MODEL
# ============================================================

print("\nTraining normal model...")

pipeline.fit(X_train, y_train)

print("Normal model training completed!")
# Save the trained model
joblib.dump(
    pipeline,
    "models/crop_yield_model.pkl"
)

print("Model saved successfully!")

# ============================================================
# 6. NORMAL MODEL PREDICTIONS
# ============================================================

print("\nMaking normal model predictions...")

y_pred = pipeline.predict(X_test)


# ============================================================
# 7. NORMAL MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n===== NORMAL MODEL EVALUATION =====")

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# ============================================================
# 8. LOG-TRANSFORMED TARGET MODEL
# ============================================================

log_model = TransformedTargetRegressor(
    regressor=pipeline,
    func=np.log1p,
    inverse_func=np.expm1
)

print("\nTraining log-transformed model...")

log_model.fit(X_train, y_train)

print("Log-transformed model training completed!")


# ============================================================
# 9. LOG MODEL PREDICTIONS
# ============================================================

print("\nMaking log-model predictions...")

log_pred = log_model.predict(X_test)


# ============================================================
# 10. LOG MODEL EVALUATION
# ============================================================

log_mae = mean_absolute_error(
    y_test,
    log_pred
)

log_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        log_pred
    )
)

log_r2 = r2_score(
    y_test,
    log_pred
)

print("\n===== LOG MODEL EVALUATION =====")

print("MAE:", log_mae)
print("RMSE:", log_rmse)
print("R2 Score:", log_r2)


# ============================================================
# 11. NORMAL-YIELD PERFORMANCE
# ============================================================

normal_mask = y_test < 100

normal_actual = y_test[normal_mask]
normal_predicted = y_pred[normal_mask]

normal_mae = mean_absolute_error(
    normal_actual,
    normal_predicted
)

normal_rmse = np.sqrt(
    mean_squared_error(
        normal_actual,
        normal_predicted
    )
)

normal_r2 = r2_score(
    normal_actual,
    normal_predicted
)

print("\n===== NORMAL-YIELD PERFORMANCE =====")

print(
    "Records with Yield < 100:",
    normal_mask.sum()
)

print("MAE:", normal_mae)
print("RMSE:", normal_rmse)
print("R2 Score:", normal_r2)


# ============================================================
# 12. LOG MODEL NORMAL-YIELD PERFORMANCE
# ============================================================

log_normal_predicted = log_pred[normal_mask]

log_normal_mae = mean_absolute_error(
    normal_actual,
    log_normal_predicted
)

log_normal_rmse = np.sqrt(
    mean_squared_error(
        normal_actual,
        log_normal_predicted
    )
)

log_normal_r2 = r2_score(
    normal_actual,
    log_normal_predicted
)

print("\n===== LOG MODEL NORMAL-YIELD PERFORMANCE =====")

print("MAE:", log_normal_mae)
print("RMSE:", log_normal_rmse)
print("R2 Score:", log_normal_r2)


# ============================================================
# 13. ACTUAL VS PREDICTED
# ============================================================

print("\n===== LOG MODEL ACTUAL VS PREDICTED =====")

results = pd.DataFrame({
    "Actual_Yield": y_test.values,
    "Predicted_Yield": log_pred
})

results["Absolute_Error"] = (
    abs(
        results["Actual_Yield"]
        - results["Predicted_Yield"]
    )
)

print(results.head(10))

print("\nLargest prediction errors:")

print(
    results
    .sort_values(
        "Absolute_Error",
        ascending=False
    )
    .head(10)
)
# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

print("\n===== FEATURE IMPORTANCE =====")

# Get the trained Random Forest
trained_model = pipeline.named_steps["model"]

# Get encoded feature names
encoded_features = (
    pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

# Get importance values
importance = trained_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": encoded_features,
    "Importance": importance
})

feature_importance = (
    feature_importance
    .sort_values("Importance", ascending=False)
)

print("\nTop 20 important features:")
print(feature_importance.head(20))
# ============================================================
# 15. TRAIN / TEST OVERLAP CHECK
# ============================================================

print("\n===== TRAIN / TEST OVERLAP CHECK =====")

train_keys = set(
    zip(
        train_data["State"],
        train_data["Crop"],
        train_data["Season"],
        train_data["Year"]
    )
)

test_keys = set(
    zip(
        test_data["State"],
        test_data["Crop"],
        test_data["Season"],
        test_data["Year"]
    )
)

overlap = train_keys.intersection(test_keys)

print("Unique train combinations:", len(train_keys))
print("Unique test combinations:", len(test_keys))
print("Overlapping combinations:", len(overlap))

print(
    "Overlap percentage:",
    len(overlap) / len(test_keys) * 100,
    "%"
)