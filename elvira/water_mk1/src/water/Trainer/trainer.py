"""
 * author Antonio Sirignano
 * created on 18-10-2026-11h-20m
 * github: https://github.com/antsiri
 * copyright 2026
"""

import pandas as pd

from tqdm import tqdm
from pathlib import Path

from water.Config.config import Config
from water.Models.models import BaseModel
from water.Validation.validation import WalkForwardValidator

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