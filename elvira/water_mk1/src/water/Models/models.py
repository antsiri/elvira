"""
 * author Antonio Sirignano
 * created on 18-10-2026-11h-06m
 * github: https://github.com/antsiri
 * copyright 2026
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

class BaseModel(ABC):
    name: str = 'base'

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        raise NotImplementedError

    @abstractmethod
    def prdict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

class PersistanceBaseline(BaseModel):
    name = 'persistance'

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
            raise NotImplementedError

class RidgeModel(BaseModel):
    name = 'ridge'

    def __init__(self, alpha: float = 1.0): ...
    def fit(self, X, y):
        raise NotImplementedError
    def predict(self, X):
        raise NotImplementedError

class RandomForestModel(BaseModel):
    name = 'random_forest'

    def __init__(self, **params): ...
    def fit(self, X, y):
        raise NotImplementedError
    def predict(self, X):
        raise NotImplementedError

class GradientBoostingModel(BaseModel):
    name = 'gradient_boosting'

    def __init__(self, **params): ...
    def fit(self, X, y):
        raise NotImplementedError
    def predict(self, X):
        raise NotImplementedError