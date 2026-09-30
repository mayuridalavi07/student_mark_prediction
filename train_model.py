import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE = Path(__file__).resolve().parent

DATA_PATH = BASE / "student_performance.csv"


TARGET = "final_exam_marks"


NUMERIC_FEATURES = [
    "study_hours",
    "attendance_percentage",
    "previous_exam_marks",
    "assignment_marks",
    "internal_marks",
    "practice_test_scores",
    "sleep_hours"
]


CATEGORICAL_FEATURES = [
    "study_method",
    "extracurricular"
]


# -------------------------------
# Load Dataset
# -------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())


# -------------------------------
# Remove Duplicates
# -------------------------------

df = df.drop_duplicates()

print("\nDataset shape after removing duplicates:")
print(df.shape)


# -------------------------------
# Separate Features and Target
# -------------------------------

X = df.drop(columns=[TARGET])

y = df[TARGET]


# -------------------------------
# Handle Outliers
# IQR Method
# -------------------------------

for column in NUMERIC_FEATURES:

    Q1 = X[column].quantile(0.25)

    Q3 = X[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR

    upper_limit = Q3 + 1.5 * IQR

    X[column] = X[column].clip(
        lower_limit,
        upper_limit
    )


# -------------------------------
# Train Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining records:", len(X_train))

print("Testing records:", len(X_test))


# -------------------------------
# Numerical Preprocessing
# -------------------------------

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


# -------------------------------
# Categorical Preprocessing
# -------------------------------

categorical_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# -------------------------------
# Column Transformer
# -------------------------------

preprocessor = ColumnTransformer(
    transformers=[

        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),

        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ]
)


# -------------------------------
# Machine Learning Model
# -------------------------------

model = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "feature_selection",
            SelectKBest(
                score_func=f_regression,
                k=8
            )
        ),

        (
            "regressor",
            LinearRegression()
        )
    ]
)


# -------------------------------
# Train Model
# -------------------------------

print("\nTraining Linear Regression model...")

model.fit(
    X_train,
    y_train
)


print("Model trained successfully!")


# -------------------------------
# Predictions
# -------------------------------

predictions = model.predict(X_test)


# -------------------------------
# Model Evaluation
# -------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    predictions
)


print("\n-----------------------------")
print("MODEL EVALUATION")
print("-----------------------------")

print("MAE :", mae)

print("MSE :", mse)

print("RMSE:", rmse)

print("R2  :", r2)


# -------------------------------
# Save Model
# -------------------------------

model_path = BASE / "student_marks_model.joblib"

joblib.dump(
    model,
    model_path
)

print("\nModel saved successfully!")

print(
    "File:",
    model_path
)


# -------------------------------
# Save Metrics
# -------------------------------

metrics = {

    "MAE": float(mae),

    "MSE": float(mse),

    "RMSE": float(rmse),

    "R2_Score": float(r2),

    "Training_Rows": len(X_train),

    "Testing_Rows": len(X_test)
}


with open(
    BASE / "metrics.json",
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


print("\nMetrics saved successfully!")

print("\nProject completed successfully!")