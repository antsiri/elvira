"""
 * author Antonio Sirignano
 * created on 18-10-2026-11h-33m
 * github: https://github.com/antsiri
 * copyright 2026
"""

import numpy as np
import pandas as pd

from water.Config.config import Config
from water.Models.models import BaseModel

class IrrigationAdvisor:
    def __init__(self, cfg: Config, model: BaseModel, volume_grid: np.ndarray | None):
        self.cfg = cfg
        self.model = model

        self.volume_grid = volume_grid

    def predict_for_volume(self, state: pd.Series, liters: float) -> float:
        raise NotImplementedError

    def recommend(self, state: pd.Series) -> dict:
        raise NotImplementedError