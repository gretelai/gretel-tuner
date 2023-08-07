from dataclasses import dataclass

import numpy as np
import pandas as pd
from category_encoders.target_encoder import TargetEncoder
from sklearn import metrics as ml_metrics
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm

__all__ = ["build_preprocessing_pipeline", "measure_ml_utility"]


@dataclass
class MLResults:
    clf: ClassifierMixin
    roc_auc: float
    precision: float
    recall: float
    f1: float
    accuracy: float

    def get_scores(self, as_dataframe=False):
        scores = {
            "roc_auc": self.roc_auc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
        }
        return pd.DataFrame(scores, index=[0]) if as_dataframe else scores


def build_preprocessing_pipeline(dataframe, target_column, cardinality_threshold=25, drop_columns=None):
    drop_columns = drop_columns or []
    features = dataframe.drop(columns=[target_column] + list(drop_columns))

    numeric_cols = features.select_dtypes(include=["number"]).columns
    categorical_cols = features.select_dtypes(include=["object", "category"]).columns

    high_cardinality_cols = [col for col in categorical_cols if features[col].nunique() > cardinality_threshold]
    categorical_cols = list(set(categorical_cols) - set(high_cardinality_cols))

    numerical_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numerical_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
            ("target_encoder", TargetEncoder(handle_missing="value", handle_unknown="value"), high_cardinality_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([("preprocessor", preprocessor)])

    return pipeline


def measure_ml_utility(
    df_real, df_holdout, target_column, df_boost=None, n_splits=3, drop_columns=None, multi_class="raise"
):
    param_grid = {
        "n_estimators": [10, 100],
        "min_samples_split": [2, 6],
        "min_samples_leaf": [1, 4],
    }
    param_scores = []
    grid_search = ParameterGrid(param_grid)
    preproc = build_preprocessing_pipeline(df_real, target_column, drop_columns=drop_columns)

    df_boost_split = None if df_boost is None else [d for d in np.array_split(df_boost.sample(frac=1), n_splits)]
    boost_desc = "" if df_boost is None else " with boosted minority"

    for params in tqdm(grid_search, total=len(grid_search), desc=f"ML utility: Performing grid search{boost_desc}"):
        cv_scores = []
        for split_i, (train_index, test_index) in enumerate(
            StratifiedKFold(n_splits=n_splits).split(df_real, df_real[target_column])
        ):
            df_train = df_real.iloc[train_index]
            df_valid = df_real.iloc[test_index]

            preproc.fit(df_train, df_train[target_column].astype("category").cat.codes.to_numpy())

            if df_boost_split is not None:
                df_train = pd.concat([df_train, df_boost_split[split_i]], ignore_index=True)

            X_train = preproc.transform(df_train)
            y_train = df_train[target_column].astype("category").cat.codes.to_numpy()

            clf = RandomForestClassifier(**params)
            clf.fit(X_train, y_train)

            X_valid = preproc.transform(df_valid)
            y_valid = df_valid[target_column].astype("category").cat.codes.to_numpy()
            y_proba = clf.predict_proba(X_valid)[:, 1] if multi_class == "raise" else clf.predict_proba(X_valid)
            cv_scores.append(ml_metrics.roc_auc_score(y_valid, y_proba, multi_class=multi_class))

        param_scores.append(np.mean(cv_scores))

    preproc.fit(df_real, df_real[target_column].astype("category").cat.codes.to_numpy())

    df_train = df_real if df_boost is None else pd.concat([df_real, df_boost], ignore_index=True)

    X_train = preproc.transform(df_train)
    y_train = df_train[target_column].astype("category").cat.codes.to_numpy()

    clf = RandomForestClassifier(**grid_search[np.argmax(param_scores)])
    clf.fit(X_train, y_train)

    X_test = preproc.transform(df_holdout)
    y_test = df_holdout[target_column].astype("category").cat.codes.to_numpy()
    y_proba = clf.predict_proba(X_test)[:, 1] if multi_class == "raise" else clf.predict_proba(X_test)
    average = "binary" if multi_class == "raise" else "weighted"

    return MLResults(
        clf=clf,
        roc_auc=ml_metrics.roc_auc_score(y_test, y_proba, multi_class=multi_class),
        f1=ml_metrics.f1_score(y_test, clf.predict(X_test), average=average),
        precision=ml_metrics.precision_score(y_test, clf.predict(X_test), average=average),
        recall=ml_metrics.recall_score(y_test, clf.predict(X_test), average=average),
        accuracy=ml_metrics.accuracy_score(y_test, clf.predict(X_test)),
    )
