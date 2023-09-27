import logging
from dataclasses import dataclass
from enum import Enum

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
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from tabulate import tabulate
from tqdm import tqdm
from xgboost import XGBClassifier

__all__ = ["build_preprocessing_pipeline", "measure_ml_utility"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PARAM_GRID_DICT = {
    "rf": {
        "n_estimators": [10, 100, 300],
        "min_samples_split": [2, 6],
        "min_samples_leaf": [1, 4],
    },
    "xgb": {
        "learning_rate": [0.01, 0.1],
        "n_estimators": [100, 200, 400],
        "max_depth": [3, 7],
        "min_child_weight": [3, 5],
    },
}


class CVMetric(str, Enum):
    PR_AUC = "pr_auc"
    ROC_AUC = "roc_auc"
    OPTIMAL_F1 = "optimal_f1"


@dataclass
class MLResults:
    clf: ClassifierMixin
    preproc: ColumnTransformer
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    hyperparameters: dict
    optimal_f1_scores: dict
    y_true: np.ndarray
    y_proba: np.ndarray

    def get_scores(self, as_dataframe=False):
        scores = {
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "optimal_f1": self.optimal_f1_scores["f1"],
            "optimal_precision": self.optimal_f1_scores["precision"],
            "optimal_recall": self.optimal_f1_scores["recall"],
            "optimal_threshold": self.optimal_f1_scores["threshold"],
        }
        return pd.DataFrame(scores, index=[0]) if as_dataframe else scores

    def print_scores(self):
        print(
            tabulate(
                self.get_scores(as_dataframe=True),
                headers="keys",
                tablefmt="grid",
                showindex=False,
                floatfmt=".2f",
            )
        )


class MLMetricCalculator:
    def __init__(self, metric, classes):
        self.metric = metric
        self.classes = classes

    def __call__(self, y_true, y_proba):
        if len(self.classes) <= 1:
            raise ValueError("MetricCalculator requires at least 2 classes")
        elif len(self.classes) == 2:
            score = self.metric(y_true, y_proba)
        else:
            scores = []
            y_true_bin = label_binarize(y_true, classes=self.classes)
            n_classes = y_true_bin.shape[1]
            for i in range(n_classes):
                scores.append(self.metric(y_true_bin[:, i], y_proba[:, i]))
            score = np.mean(scores)
        return score


def optimize_f1(y_test, y_proba, return_param="f1"):
    precision, recall, thresholds = ml_metrics.precision_recall_curve(y_test, y_proba)
    f1_array = (2 * precision * recall) / (precision + recall + 1e-8)
    index = np.argmax(f1_array)
    results = {
        "f1": f1_array[index],
        "precision": precision[index],
        "recall": recall[index],
        "threshold": thresholds[index],
    }
    return results[return_param] if return_param != "all" else results


def pr_auc_score(y_test, y_proba):
    precision, recall, _ = ml_metrics.precision_recall_curve(y_test, y_proba)
    return ml_metrics.auc(recall, precision)


def build_preprocessing_pipeline(dataframe, target_column, cardinality_threshold=25, drop_columns=None):
    drop_columns = drop_columns or []

    logger.info(f"ML utility - Dropping columns: {drop_columns}")
    features = dataframe.drop(columns=[target_column] + list(drop_columns))

    numeric_cols = features.select_dtypes(include=["number"]).columns
    categorical_cols = features.select_dtypes(include=["object", "category"]).columns

    high_cardinality_cols = [col for col in categorical_cols if features[col].nunique() > cardinality_threshold]
    categorical_cols = list(set(categorical_cols) - set(high_cardinality_cols))

    numerical_pipeline = Pipeline([("scaler", StandardScaler()), ("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]
    )

    transformers = [
        ("numeric", numerical_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ]

    if len(high_cardinality_cols) > 0:
        transformers.append(
            ("target_encoder", TargetEncoder(handle_missing="value", handle_unknown="value"), high_cardinality_cols)
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def measure_ml_utility(
    df_real,
    df_holdout,
    target_column,
    df_boost=None,
    n_splits=3,
    drop_columns=None,
    is_multi_class=False,
    param_grid=None,
    classifier="rf",
    cv_metric=CVMetric.PR_AUC,
):
    df_ref = df_real.copy()

    Classifier = {"rf": RandomForestClassifier, "xgb": XGBClassifier}[classifier]
    param_grid = param_grid or PARAM_GRID_DICT[classifier]

    preproc = build_preprocessing_pipeline(df_ref, target_column, drop_columns=drop_columns)
    df_boost_split = None if df_boost is None else [d for d in np.array_split(df_boost.sample(frac=1), n_splits)]

    multi_class = "ovr" if is_multi_class else "raise"
    score_average = "weighted" if is_multi_class else "binary"
    auc_average = "weighted" if is_multi_class else "macro"
    boost_desc = "" if df_boost is None else " with boosted minority"

    param_scores = []
    grid_search = ParameterGrid(param_grid)
    skf = StratifiedKFold(n_splits=n_splits)

    logger.info(f"ML utility - Training {classifier.upper()} model using {cv_metric} for cross-validation")

    cv_metric_func = {
        CVMetric.ROC_AUC: ml_metrics.roc_auc_score,
        CVMetric.PR_AUC: pr_auc_score,
        CVMetric.OPTIMAL_F1: optimize_f1,
    }[cv_metric]

    classes = df_ref[target_column].unique()
    cv_metric_func = MLMetricCalculator(metric=cv_metric_func, classes=classes)

    for params in tqdm(grid_search, total=len(grid_search), desc=f"ML utility - Performing grid search{boost_desc}"):
        cv_scores = []
        for split_i, (train_index, test_index) in enumerate(skf.split(df_ref, df_ref[target_column])):
            df_train = df_ref.iloc[train_index]
            df_valid = df_ref.iloc[test_index]

            preproc.fit(df_train, df_train[target_column].astype("category").cat.codes.to_numpy())

            if df_boost_split is not None:
                df_train = pd.concat([df_train, df_boost_split[split_i]], ignore_index=True)

            X_train = preproc.transform(df_train)
            y_train = df_train[target_column].astype("category").cat.codes.to_numpy()

            clf = Classifier(**params)
            clf.fit(X_train, y_train)

            X_valid = preproc.transform(df_valid)
            y_valid = df_valid[target_column].astype("category").cat.codes.to_numpy()
            y_proba = clf.predict_proba(X_valid) if is_multi_class else clf.predict_proba(X_valid)[:, 1]

            cv_scores.append(cv_metric_func(y_valid, y_proba))

        param_scores.append(np.mean(cv_scores))

    best_params = grid_search[np.argmax(param_scores)]
    logger.info(
        f"ML utility - Grid search complete -> {cv_metric} hyperparameters:\n"
        + tabulate(pd.DataFrame(best_params, index=[0]), headers="keys", tablefmt="grid", showindex=False)
        + "\n"
    )

    preproc.fit(df_ref, df_ref[target_column].astype("category").cat.codes.to_numpy())

    df_train = df_ref if df_boost is None else pd.concat([df_ref, df_boost], ignore_index=True)
    logger.info(f"ML utility - Final training on {len(df_train)} samples")
    logger.info(f"ML utility - len(real) = {len(df_ref)}, len(boost) = {len([] if df_boost is None else df_boost)}")

    X_train = preproc.transform(df_train)
    y_train = df_train[target_column].astype("category").cat.codes.to_numpy()

    clf = Classifier(**best_params)
    clf.fit(X_train, y_train)

    X_test = preproc.transform(df_holdout)
    y_test = df_holdout[target_column].astype("category").cat.codes.to_numpy()

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test) if is_multi_class else clf.predict_proba(X_test)[:, 1]

    logger.info(
        "ML utility - Classification Report:\n"
        f"{ml_metrics.classification_report(y_test, y_pred, target_names=classes)}"
    )

    return MLResults(
        clf=clf,
        preproc=preproc,
        roc_auc=ml_metrics.roc_auc_score(y_test, y_proba, multi_class=multi_class, average=auc_average),
        f1=ml_metrics.f1_score(y_test, y_pred, average=score_average),
        precision=ml_metrics.precision_score(y_test, y_pred, average=score_average),
        recall=ml_metrics.recall_score(y_test, y_pred, average=score_average),
        accuracy=ml_metrics.accuracy_score(y_test, y_pred),
        hyperparameters=best_params,
        pr_auc=pr_auc_score(y_test, y_proba),
        optimal_f1_scores=optimize_f1(y_test, y_proba, return_param="all"),
        y_true=y_test,
        y_proba=y_proba,
    )
