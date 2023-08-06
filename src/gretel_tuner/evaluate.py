import logging
import time
from typing import Callable

from gretel_client.evaluation.downstream_classification_report import DownstreamClassificationReport
from tabulate import tabulate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


__all__ = ["evaluate_and_print_average_score"]


def _validate_data(df, target):
    """Validates if the 'target' column exists in the DataFrame."""
    if target not in df.columns:
        raise ValueError(f"The 'target' column '{target}' does not exist in the data.")


def calculate_scores(evaluate: dict, metric: str, score_agg_func: Callable = max):
    """Calculate evaluation scores and log the results.

    Parameters:
        evaluate (dict): The evaluation results returned by DownstreamClassificationReport.
        metric (str): The evaluation metric to be used (e.g., 'f1', 'precision', 'recall', 'auc').
        score_agg_func (callable): Aggregation function to be used to calculate the score (default: max).

    Returns:
        dict: A dictionary containing the evaluation metric and its score for both 'synth_scores' and 'train_scores'.
    """
    metric = metric.upper()
    data_types = ["synth_scores", "train_scores"]
    log_table = [["Model", f"{metric} Score - synth", f"{metric} Score - real world"]]

    for model in evaluate.as_dict[data_types[0]][metric].keys():  # iterate over models
        synth_score = evaluate.as_dict[data_types[0]][metric][model]
        training_score = evaluate.as_dict[data_types[1]][metric][model]
        log_table.append([model, synth_score, training_score])

    logger.info("\n" + tabulate(log_table, headers="firstrow", tablefmt="grid"))

    # calculate average or max score for each data type
    result_scores = {}
    for data_type in data_types:
        metric_scores = evaluate.as_dict[data_type][metric].values()
        metric_score = score_agg_func(metric_scores)
        logger.info(f"{data_type}: {score_agg_func} {metric} Score: {metric_score}")
        result_scores[data_type] = {metric: metric_score}

    return result_scores


def evaluate_and_print_average_score(
    project,
    df_real,
    df_synth,
    target,
    test_holdout=0.2,
    models=["knn", "dt", "svm", "rf"],
    metric="f1",
    record_count=1e6,
    score_agg_func=max,
):
    """Evaluate synthetic data using downstream classification models and print the average score.

    Parameters:
        df_real (pd.DataFrame): The real (reference) DataFrame.
        df_synth (pd.DataFrame): The synthetic DataFrame to be evaluated.
        target (str): The name of the target column.
        test_holdout (float): The percentage of data to be used for testing.
        models (list): The list of classification models to evaluate (default: ["knn", "dt", "svm", "rf"]).
        metric (str): The evaluation metric to be used (default: "fx1").
        record_count (float): The number of records to use for evaluation (default: 1e6).
        score_agg_func (callable): Aggregation function to be used to calculate the score (default: max).

    Returns:
        dict: A dictionary containing the evaluation metric and its score.
    """
    _validate_data(df_synth, target)

    evaluate = DownstreamClassificationReport(
        project=project,
        target=target,
        data_source=df_synth,
        ref_data=df_real,
        holdout=test_holdout,
        models=models,
        metric=metric,
        runner_mode="cloud",
        record_count=record_count,
    )

    logger.info(f"Starting evaluation with {len(df_real)} real records and {len(df_synth)} synthetic records.")
    start_time = time.time()
    evaluate.run()
    duration = time.time() - start_time
    logger.info(f"Evaluation completed in {duration:.2f} seconds.")

    return calculate_scores(evaluate, metric, score_agg_func)
