import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import gretel_client as gretel
import optuna

from .metrics import BaseTunerMetric

__all__ = ["GretelHyperParameterConfig", "GretelHyperParameterTuner"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GretelHyperParameterConfig:
    project: gretel.projects.Project
    artifact_id: str
    metric: BaseTunerMetric
    base_config: dict = field(
        default_factory=lambda: gretel.projects.models.read_model_config("synthetics/tabular-actgan")
    )
    epoch_choices: List[int] = field(default_factory=lambda: [100, 200, 400, 600, 800, 1000])
    batch_size_choices: List[int] = field(default_factory=lambda: [1000])
    layer_width_choices: List[int] = field(default_factory=lambda: [512, 1024, 2048])
    num_layers_range: Tuple[int, int] = (2, 4)
    generator_lr_range: Tuple[float, float] = (1e-5, 1e-3)
    discriminator_lr_range: Tuple[float, float] = (1e-5, 1e-3)


@dataclass(frozen=True)
class GretelHyperParameterResults:
    best_config: dict
    optuna_study: optuna.Study


class GretelHyperParameterTuner:
    def __init__(self, tuner_config: GretelHyperParameterConfig):
        self.tuner_config = tuner_config
        self.project = tuner_config.project
        self.artifact_id = tuner_config.artifact_id
        self.runner_mode = self.project.client_config.default_runner
        self.artifact_endpoint = self.project.client_config.artifact_endpoint
        self.metric = tuner_config.metric
        self.__base_config = deepcopy(tuner_config.base_config)

    def _get_trial_config(self, trial: optuna.Trial):
        return self.get_config(
            epochs=trial.suggest_categorical("epochs", choices=self.tuner_config.epoch_choices),
            batch_size=trial.suggest_categorical("batch_size", choices=self.tuner_config.batch_size_choices),
            generator_lr=trial.suggest_float("generator_lr", *self.tuner_config.generator_lr_range, log=True),
            discriminator_lr=trial.suggest_float(
                "discriminator_lr", *self.tuner_config.discriminator_lr_range, log=True
            ),
            num_layers=trial.suggest_int("num_layers", *self.tuner_config.num_layers_range),
            layer_width=trial.suggest_categorical("layer_width", choices=self.tuner_config.layer_width_choices),
        )

    def get_config(
        self,
        epochs: int,
        batch_size: int,
        generator_lr: float,
        discriminator_lr: float,
        num_layers: int,
        layer_width: int,
    ):
        c = deepcopy(self.__base_config)
        c["models"][0]["actgan"]["params"]["epochs"] = epochs
        c["models"][0]["actgan"]["params"]["batch_size"] = batch_size
        c["models"][0]["actgan"]["params"]["generator_lr"] = generator_lr
        c["models"][0]["actgan"]["params"]["discriminator_lr"] = discriminator_lr
        c["models"][0]["actgan"]["params"]["generator_dim"] = [layer_width for _ in range(num_layers)]
        c["models"][0]["actgan"]["params"]["discriminator_dim"] = [layer_width for _ in range(num_layers)]
        return c

    def objective(self, trial: optuna.Trial):
        trial_config = self._get_trial_config(trial)

        try:
            trial_config["name"] = f"{trial_config['name']}-optuna-{trial.number}"
            model = self.project.create_model_obj(trial_config, data_source=self.artifact_id)
            model.submit()
            gretel.helpers.poll(model, verbose=False)
            return self.metric(model)

        except Exception as e:
            raise Exception(f"Model run failed with error {e}, using config {trial_config}")

    def run(
        self, n_trials: int = 10, n_jobs: int = 1, study: Optional[optuna.Study] = None, **kwargs
    ) -> GretelHyperParameterResults:
        if study is None:
            study = optuna.create_study(study_name=f"optuna-study_{self.project.name}", direction="maximize")
        study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs, **kwargs)
        return GretelHyperParameterResults(best_config=self.get_config(**study.best_params), optuna_study=study)
