import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import gretel_client as gretel
import pandas as pd
import smart_open
from sdmetrics.reports.single_table import QualityReport
from tabulate import tabulate

from .ml import measure_ml_utility

__all__ = [
    "BaseTunerMetric",
    "GretelSQSMetric",
    "SDMetricsScore",
    "BinaryMinorityBoostMetric",
    "MLUtilityMetric",
    "MultiClassMinorityBoostMetric",
]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTunerMetric(ABC):
    def __init__(self, client=None):
        self.client = client

    @property
    def name(self):
        return self.__class__.__name__

    def _get_gretel_report(self, model: gretel.projects.models.Model):
        if self.client is None and "azure:" in model.project.client_config.artifact_endpoint:
            raise Exception("A client object must be provided at initialization for Azure hybrid runs.")
        tp = {"client": self.client} if self.client else {}
        with smart_open.open(model.get_artifact_link("report_json"), mode="rb", transport_params=tp) as json_file:
            report = json.load(json_file)
        return report

    @abstractmethod
    def __call__(self, model: gretel.projects.models.Model) -> float:
        """Calculate the optimization metric and return the score as a float."""


class GretelSQSMetric(BaseTunerMetric):
    def __call__(self, model: gretel.projects.models.Model) -> float:
        return self._get_gretel_report(model)["synthetic_data_quality_score"]["raw_score"]


class SDMetricsScore(BaseTunerMetric):
    _gretel_to_sdtype = {
        "numeric": "numerical",
        "categorical": "categorical",
        "binary": "boolean",
        "other": "categorical",
    }

    def __init__(self, df_real: pd.DataFrame, client: Optional[Any] = None, metadata: Optional[Dict[str, str]] = None):
        super().__init__(client)
        self.df_real = df_real.copy()
        self.metadata = metadata

    def _get_metadata(self, model: gretel.projects.models.Model):
        gretel_report = self._get_gretel_report(model)
        metadata = {
            "columns": {
                f["name"]: {"sdtype": self._gretel_to_sdtype[f["left_field_features"]["type"]]}
                for f in gretel_report["fields"]
            }
        }
        return metadata

    def _get_synthetic_data(self, model: gretel.projects.models.Model) -> pd.DataFrame:
        tp = {"client": self.client} if self.client else {}
        with smart_open.open(model.get_artifact_link("data_preview"), mode="rb", transport_params=tp) as file_in:
            df_synth = pd.read_csv(file_in)
        return df_synth

    def __call__(self, model: gretel.projects.models.Model):
        self.metadata = self.metadata or self._get_metadata(model)
        sdmetrics_report = QualityReport()
        df_synth = self._get_synthetic_data(model)
        sdmetrics_report.generate(self.df_real, df_synth, self.metadata)
        return sdmetrics_report.get_score()


class MLUtilityMetric:
    def __init__(self, df_test, target_column, ml_metric="pr_auc", drop_columns=None, classifier="rf"):
        self.df_test = df_test
        self.ml_metric = ml_metric
        self.drop_columns = drop_columns
        self.classifier = classifier
        self.target_column = target_column

    def __call__(self, model: gretel.projects.models.Model):
        df_synth = pd.read_csv(model.get_artifact_link("data_preview"), compression="gzip")

        results = measure_ml_utility(
            df_real=df_synth,
            df_holdout=self.df_test,
            target_column=self.target_column,
            cv_metric=self.ml_metric,
            classifier=self.classifier.lower(),
        )

        return results.get_scores()[self.ml_metric]


class MinorityBoostingMetric(BaseTunerMetric):
    is_multi_class = False

    def __init__(self, df_train, df_holdout, target_column, ml_metric="roc_auc", drop_columns=None, client=None):
        self.df_train = df_train
        self.df_holdout = df_holdout
        self.target_column = target_column
        self.ml_metric = ml_metric
        self.drop_columns = drop_columns
        self.results_no_boost = None
        super().__init__(client)

    def _get_synthetic_data(self, rh: gretel.projects.records.RecordHandler) -> pd.DataFrame:
        tp = {"client": self.client} if self.client else {}
        with smart_open.open(rh.get_artifact_link("data"), mode="rb", transport_params=tp) as file_in:
            df_synth = pd.read_csv(file_in)
        return df_synth

    @abstractmethod
    def generate_synthetic_minority(self, model: gretel.projects.models.Model) -> pd.DataFrame:
        ...

    def __call__(self, model):
        df_synth = self.generate_synthetic_minority(model)

        ml_utility_kwargs = dict(
            df_real=self.df_train,
            df_holdout=self.df_holdout,
            target_column=self.target_column,
            drop_columns=self.drop_columns,
            is_multi_class=self.is_multi_class,
        )

        if self.results_no_boost is None:
            self.results_no_boost = measure_ml_utility(df_boost=None, **ml_utility_kwargs)

        results_boosted = measure_ml_utility(df_boost=df_synth, **ml_utility_kwargs)

        df_results = pd.concat(
            [self.results_no_boost.get_scores(as_dataframe=True), results_boosted.get_scores(as_dataframe=True)],
            ignore_index=True,
        )
        df_results["with synths"] = ["no", "yes"]

        logger.info("\n" + tabulate(df_results, headers="keys", tablefmt="grid", showindex=False))

        score = results_boosted.get_scores()[self.ml_metric.lower()]
        logger.info(f"Optimization metric score: {self.ml_metric} = {score:.4f}\n")

        return score


class BinaryMinorityBoostMetric(MinorityBoostingMetric):
    def __init__(
        self,
        df_train,
        df_holdout,
        target_column,
        minority_class,
        boost_number,
        ml_metric="roc_auc",
        drop_columns=None,
        client=None,
    ):
        super().__init__(
            df_train=df_train,
            df_holdout=df_holdout,
            target_column=target_column,
            ml_metric=ml_metric,
            drop_columns=drop_columns,
            client=client,
        )
        self.boost_number = boost_number
        self.minority_class = minority_class

    def generate_synthetic_minority(self, model: gretel.projects.models.Model) -> pd.DataFrame:
        seeds = pd.DataFrame(data=[self.minority_class] * self.boost_number, columns=[self.target_column])
        record_handler = model.create_record_handler_obj(data_source=seeds, params={"num_records": len(seeds)})
        record_handler.submit()
        gretel.helpers.poll(record_handler, verbose=False)
        return self._get_synthetic_data(record_handler)


class MultiClassMinorityBoostMetric(MinorityBoostingMetric):
    is_multi_class = True

    def __init__(
        self,
        df_train,
        df_holdout,
        target_column,
        minority_classes,
        boost_numbers,
        ml_metric="roc_auc",
        drop_columns=None,
        client=None,
    ):
        super().__init__(
            df_train=df_train,
            df_holdout=df_holdout,
            target_column=target_column,
            ml_metric=ml_metric,
            drop_columns=drop_columns,
            client=client,
        )
        self.boost_numbers = boost_numbers
        self.minority_classes = minority_classes

    def generate_synthetic_minority(self, model: gretel.projects.models.Model) -> pd.DataFrame:
        seeds = pd.DataFrame(
            data=sum([[c] * num for c, num in zip(self.minority_classes, self.boost_numbers)], []),
            columns=[self.target_column],
        )
        record_handler = model.create_record_handler_obj(data_source=seeds, params={"num_records": len(seeds)})
        record_handler.submit()
        gretel.helpers.poll(record_handler, verbose=False)
        return self._get_synthetic_data(record_handler)
