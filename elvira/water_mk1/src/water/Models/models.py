"""
 * author Antonio Sirignano
 * created on 18-10-2026-11h-06m
 * github: https://github.com/antsiri
 * copyright 2026
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from typing import Any

import numpy as np
import pandas as pd

from water.Config.config import Config
class BaseModel(ABC):
    base_name: str = 'base'
    DEFAULTS: dict[str, Any] = {}

    def __init__(self, **params: Any):
        self.params: dict[str, Any] = {**self.DEFAULTS, **params}
        self.feature_names_: list[str] | None = None

    @property
    def name(self) -> str:
        changed = {k: v for k, v in self.params.items() if self.DEFAULTS.get(k) != v}
        if not changed:
            return self.base_name
        return f'{self.base_name}({', '.join(f'{k}: {v}' for k, v in changed.items())})'

    def clone(self) -> 'BaseModel':
        return type(self)(**self.params)

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel': ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def feature_weights(self) -> pd.Series | None:
        return None

    def partial_fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        raise NotImplementedError(f'{self.name} doesn\'t support partial_fit')

    def __repr__(self) -> str:
        return f'<{self.name}>'


class PersistanceBaseline(BaseModel):
    base_name = 'persistance'
    DEFAULTS = {'column': 'wp_t'}

    def fit(self, X: pd.Series, y: pd.Series) -> 'PersistanceBaseline':
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        col = self.params['column']
        if col not in X.columns:
            raise KeyError(
                f'[PersistanceBaseline] It is necessart the column "{col}".'
            )
        return X[col].to_numpy(dtype=float)

class SklearnModel(BaseModel):

    @abstractmethod
    def _build_estimator(self): ...

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'SklearnModel':
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame")
        self.feature_names_ = list(X.columns)
        self.estimator_.fit(X[self.feature_names_], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.feature_names_ is None:
            raise RuntimeError(f'{self.name}: run fit() before predict()')
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise KeyError(f'{self.name}: missing columns in predict: {missing}')
        return self.estimator_.predict(X[self.feature_names_])

class RidgeModel(SklearnModel):
    base_name = 'ridge'
    DEFAULTS = {'alpha': 1.0}

    def _build_estimator(self):
        return make_pipeline(StandardScaler(), Ridge(alpha=self.params['alpha']))
    
    def feature_weights(self) -> pd.Series | None:
        if self.feature_weights is None:
            return None
        coef = self.estimator_[-1].coef_
        return pd.Series(coef, index=self.feature_names_).sort_values(key=np.abs, ascending=False)

class RandomForestModel(SklearnModel):
    base_name = 'random_forest'
    DEFAULTS = {
        'n_estimatots': 300, 'max_depth': 4, 'min_samples_leaf': 5,
        'max_features': 0.5, 'random_state': 42, 'n_jobs': -1
    }

    def _build_estimator(self):
        return RandomForestRegressor(**self.params)

    def feature_weights(self) -> pd.Series | None:
        if self.feature_names_ is None:
            return None
        return pd.Series(self.estimator_.feature_importance_, index=self.feature_names_).sort_values(ascending=False)
    

class GradientBoostingModel(SklearnModel):
    base_name = 'gradient_boosting'
    DEFAULTS = {
        'n_estimatots': 200, 'learning_rate': 0.05, 'max_depth': 2,
        'subsamples': 0.8, 'min_samples_leaf': 5, 'random_state': 42
        }

    def _build_estimator(self):
        return GradientBoostingRegressor(**self.params)

    def feature_weights(self) -> pd.Series | None:
        if self.feature_names_ is None:
            return None
        return pd.Series(self.estimator_.feature_importance_, index=self.feature_names_).sort_values(ascending=False)


class DeltaModel(BaseModel):
    base_name = 'delta'

    def __init__(self, inner: BaseModel, wp_is_features: bool = True, column: str= 'wp_t'):
        super().__init__(inner=inner, wp_is_features=wp_is_features, column=column)
        self.inner = inner

    @property
    def name(self) -> str:
        suffix = '' if self.params['wp_is_features'] else ', without wp_t'
        return f'delta[{self.inner.name}{suffix}]'

    def clone(self) -> 'DeltaModel':
        return DeltaModel(self.inner.clone(), self.params['wp_is_feature'], self.params['column'])

    def _check(self, X: pd.DataFrame) -> str:
        col = self.params['column']
        if col not in X.columns:
            raise KeyError(f'{self.name}: it is necessary the columns "{col}" in X to the reconstruction level')
        return col

    def _inner_X(self, X: pd.DataFrame) -> pd.DataFrame:
        col = self._check(X)
        return X if self.params['wp_is_features'] else X.drop(columns=col)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'DeltaModel':
        col = self._check(X)
        self.inner.fit(self._inner_X(X), y - X[col])
        self.feature_names_ = self.inner.feature_names_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        col = self._check(X)
        return X[col].to_numpy(dtype=float) + self.inner.predict(self._inner_X(X))

    def feature_weights(self) -> pd.Series:
        return self.inner.feature_weights()


def default_models(cfg: Config) -> list[BaseModel]:
    rs = cfg.random_state
    return [
        PersistanceBaseline(),
        RidgeModel(alpha=1.0),
        RidgeModel(alpha=10.0),
        RidgeModel(alpha=100.0),
        RandomForestModel(random_state=rs),
        GradientBoostingModel(random_state=rs),
    ]