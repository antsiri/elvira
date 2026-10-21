"""
 * author Antonio Sirignano
 * created on 18-10-2026-11h-20m
 * github: https://github.com/antsiri
 * copyright 2026
"""

from __future__ import annotations

import joblib
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from itertools import product

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from water.Config.config import Config
from water.Models.models import BaseModel
from water.Validation.validation import WalkForwardValidator

WP_COL = 'wp_t'
Y_COL = 'y'
@dataclass(frozen=True)
class Variant:
    use_wp_t: bool
    delta: bool

    @property
    def label(self) -> str:
        return f'{'delta' if self.delta else 'level'} | {'with' if self.use_wp_t else 'without'} wp_t'

class Evaluator:
    @staticmethod
    def metrics(y_true, y_pred) -> dict:
        raise NotImplementedError

class Trainer:
    def __init__(self, cfg: Config, models: list[BaseModel], validator: WalkForwardValidator):
        self.cfg = cfg
        self.models = models
        self.validator = validator

        self.results_ = pd.DataFrame | None = None

    def cross_validate(self, df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        raise NotImplementedError

    def fit_final(self, model: BaseModel, X: pd.DataFrame, y: pd.Series) -> BaseModel:
        raise NotImplementedError

    def save(self, model: BaseModel, path: Path):
        raise NotImplementedError