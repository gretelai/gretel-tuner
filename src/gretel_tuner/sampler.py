from abc import abstractmethod
from copy import deepcopy
from typing import List, Optional, Tuple

import optuna
from gretel_client.projects.models import read_model_config
from pydantic import BaseModel, Field

__all__ = ["BaseConfigSampler", "ACTGANConfigSampler", "GPTXConfigSampler"]


class BaseConfigSampler(BaseModel):
    """Base class for config samplers."""

    base_model_config: dict

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    @abstractmethod
    def _get_trial_config(self, trial: optuna.Trial) -> dict:
        """Return a model config dict for a single Optuna trial."""
        ...

    @abstractmethod
    def create_config(self, **kwargs) -> dict:
        """Return a model config given specific sampler parameter values."""
        ...


class ACTGANConfigSampler(BaseConfigSampler):
    base_model_config: dict = Field(default=read_model_config("synthetics/tabular-actgan"))
    epochs_choices: List[int] = Field(default=[100, 200, 400, 800], gt=0)
    batch_size_choices: List[int] = Field(default=[1000], gt=0)
    layer_width_choices: List[int] = Field(default=[512, 1024, 2048], gt=0)
    num_layers_range: Tuple[int, int] = Field(default=(2, 4), gt=0)
    generator_lr_range: Tuple[float, float] = Field(default=(1e-5, 1e-3), gt=0)
    discriminator_lr_range: Tuple[float, float] = Field(default=(1e-5, 1e-3), gt=0)

    def _get_trial_config(self, trial: optuna.Trial) -> dict:
        return self.create_config(
            epochs=trial.suggest_categorical("epochs", choices=self.epochs_choices),
            batch_size=trial.suggest_categorical("batch_size", choices=self.batch_size_choices),
            generator_lr=trial.suggest_float("generator_lr", *self.generator_lr_range, log=True),
            discriminator_lr=trial.suggest_float("discriminator_lr", *self.discriminator_lr_range, log=True),
            num_layers=trial.suggest_int("num_layers", *self.num_layers_range),
            layer_width=trial.suggest_categorical("layer_width", choices=self.layer_width_choices),
        )

    def create_config(
        self,
        epochs: int,
        batch_size: int,
        generator_lr: float,
        discriminator_lr: float,
        num_layers: int,
        layer_width: int,
    ) -> dict:
        c = deepcopy(self.base_model_config)
        c["models"][0]["actgan"]["params"]["epochs"] = epochs
        c["models"][0]["actgan"]["params"]["batch_size"] = batch_size
        c["models"][0]["actgan"]["params"]["generator_lr"] = generator_lr
        c["models"][0]["actgan"]["params"]["discriminator_lr"] = discriminator_lr
        c["models"][0]["actgan"]["params"]["generator_dim"] = [layer_width for _ in range(num_layers)]
        c["models"][0]["actgan"]["params"]["discriminator_dim"] = [layer_width for _ in range(num_layers)]
        return c


def _check_steps_epochs(steps: Optional[int], epochs: Optional[int]) -> None:
    if steps is None and epochs is None:
        raise ValueError("Either steps or epochs must be specified")
    elif steps is not None and epochs is not None:
        raise ValueError("Only one of steps or epochs can be specified")


class GPTXConfigSampler(BaseConfigSampler):
    base_model_config: dict = Field(default=read_model_config("synthetics/natural-language"))
    epochs_choices: Optional[List[int]] = Field(default=None, gt=0)
    steps_choices: Optional[List[int]] = Field(default=[650, 700, 750, 800, 850, 900], gt=0)
    batch_size_choices: List[int] = Field(default=[4], gt=0)
    warmup_steps_choices: List[int] = Field(default=[100, 200], gt=0)
    max_tokens_choices: List[int] = Field(default=[512], gt=0)
    lr_scheduler_choices: List[str] = Field(default=["linear"])
    learning_rate_range: Tuple[float, float] = Field(default=(1e-5, 1e-3), gt=0)
    weight_decay_range: Tuple[float, float] = Field(default=(0.001, 0.05), gt=0)

    def _get_trial_config(self, trial: optuna.Trial) -> dict:
        epoch_trials = trial.suggest_categorical("epochs", choices=self.epochs_choices) if self.epochs_choices else None
        steps_trials = trial.suggest_categorical("steps", choices=self.steps_choices) if self.steps_choices else None
        _check_steps_epochs(steps_trials, epoch_trials)
        return self.create_config(
            epochs=epoch_trials,
            steps=steps_trials,
            batch_size=trial.suggest_categorical("batch_size", choices=self.batch_size_choices),
            warmup_steps=trial.suggest_categorical("warmup_steps", choices=self.warmup_steps_choices),
            max_tokens=trial.suggest_categorical("max_tokens", choices=self.max_tokens_choices),
            lr_scheduler=trial.suggest_categorical("lr_scheduler", choices=self.lr_scheduler_choices),
            learning_rate=trial.suggest_float("learning_rate", *self.learning_rate_range, log=True),
            weight_decay=trial.suggest_float("weight_decay", *self.weight_decay_range, log=True),
        )

    def create_config(
        self,
        batch_size: int,
        warmup_steps: int,
        max_tokens: int,
        lr_scheduler: str,
        learning_rate: float,
        weight_decay: float,
        epochs: Optional[int] = None,
        steps: Optional[int] = None,
    ) -> dict:
        c = deepcopy(self.base_model_config)
        _check_steps_epochs(steps, epochs)
        c["models"][0]["gpt_x"]["steps"] = steps
        c["models"][0]["gpt_x"]["epochs"] = epochs
        c["models"][0]["gpt_x"]["batch_size"] = batch_size
        c["models"][0]["gpt_x"]["warmup_steps"] = warmup_steps
        c["models"][0]["gpt_x"]["max_tokens"] = max_tokens
        c["models"][0]["gpt_x"]["lr_scheduler"] = lr_scheduler
        c["models"][0]["gpt_x"]["learning_rate"] = learning_rate
        c["models"][0]["gpt_x"]["weight_decay"] = weight_decay
        return c
