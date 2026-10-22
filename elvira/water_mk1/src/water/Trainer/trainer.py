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
from water.Models.models import BaseModel, DeltaModel, PersistanceBaseline
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

DEFAULTS_VARIANTS = (
    Variant(use_wp_t=False, delta=False),
    Variant(use_wp_t=False, delta=True),
    Variant(use_wp_t=True, delta=False),
    Variant(use_wp_t=True, delta=True),
)

class Evaluator:
    @staticmethod
    def metrics(y_true, y_pred) -> dict:
        y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
        return {
            'n': len(y_true),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2': r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
        }

class Trainer:
    def __init__(self, cfg: Config, models: list[BaseModel], validator: WalkForwardValidator, variants: tuple[Variant, ...] = DEFAULTS_VARIANTS):
        self.cfg = cfg
        self.validator = validator
        self.variants = variants

        self.models = [m for m in models if not isinstance(m, PersistanceBaseline)]
        self.baseline = PersistanceBaseline()
        self.results_: pd.DataFrame | None = None
        self.predictions_ = pd.DataFrame | None = None

#   HELPER -----------------------------------------------------------------------------------------

    @staticmethod
    def _base_features(features: list[str]) -> list[str]:
        return [c for c in features if c != WP_COL]

    def _columns(self, features: list[str], variant: Variant) -> list[str]:
        cols = self._base_features(features)
        return cols + [WP_COL] if (variant.use_wp_t or variant.delta) else cols

    @staticmethod
    def _wrap(model: BaseModel, variant: Variant) -> BaseModel:
        m = model.clone()
        return DeltaModel(m, wp_is_features=variant.use_wp_t) if variant.delta else m

#   WALK-FORWARD -----------------------------------------------------------------------------------

    def cross_validate(self, df: pd.DataFrame, features: list [str], zones: tuple[int | None, ...] = (0, 1)) -> pd.DataFrame:
        jobs: list[tuple] = []
        for z in zones:
            jobs.append((z, None, self.baseline))
            jobs += [(z, v, m) for v, m in product(self.variants, self.models)]

        rows, preds = [], []
        for z, variant, model in tqdm(jobs, desc='Evaluation', unit='model'):
            vlabel = 'baseline' if variant is None else self._columns(features, variant)
            cols = [WP_COL] if variant is None else self._columns(features, variant)
            zlabel = 'all' if z is None else z
            folds = list(self.validator.split(df, z))
            for k, (tr, te) in enumerate(tqdm(folds, desc='\t{model.name}', leave=False)):
                est = model.clone() if variant is None else self._wrapo(model, variant)
                Xtr, Xte = df[cols].iloc[tr], df[cols].iloc[te]
                est.fit(Xtr, df[Y_COL].iloc[tr])
                yhat = est.predict(Xte)
                ytrue = df[Y_COL].iloc[te].to_numpy()
                rows.append({
                    'zone': zlabel,
                    'variant': vlabel,
                    'model': est.name if variant else model.name,
                    'fold': k,
                    **Evaluator.metrics(ytrue, yhat)
                })
                preds.append(pd.DataFrame({
                    'row': df.index[te],
                    'giorno': df[self.cfg.date_col].iloc[te].to_numpy(),
                    'zone': zlabel,
                    variant: vlabel,
                    'fold': k,
                    'model': est.name if variant else model.name,
                    'y_true': ytrue,
                    'y_pred': yhat
                }))

        self.results_ = pd.DataFrame(rows)
        self.predictions_ = pd.concat(preds, ignore_index=True)
        return self.summary()

    def summary(self) -> pd.DataFrame:
        if self.predictions_ is None:
            raise RuntimeError('Run cross_validate() before.')
        p, r = self.predictions_, self.results_
        out = []
        for (z,  v, m), g in p.groupby(['zone', 'variant', 'model'], sort=False):
            met = Evaluator.metrics(g.y_true, g.y_pred)
            fold_std = r[(r.zone == z) & (r.varaint == v) & (r.model == m)].mae.std()
            out.append({
                'zone': z,
                'varaint': v,
                'model': m,
                **met,
                'fold_mae_std': fold_std
            })
        s = pd.DataFrame(out)
        base = s[s.variant == 'baseline'].set_index('zone').mae
        s['skill_vs_persistance'] = 1 - s.mae / s.zone.map(base)
        return s.sort_values(['zone', 'mae']).reset_index(drop=True)

#   PREQUENTIAL ------------------------------------------------------------------------------------

    def prequential(self, df: pd.DataFrame, features: list[str],
                    model: BaseModel, variant: Variant, zone: int | None = None,
                    refit_every: int = 1,
                    ) -> pd.DataFrame:
        cfg = self.cfg
        cols = self._columns(features, variant)
        dates = pd.DatetimeIndex(df[cfg.date_col])
        zones = df[cfg.date_col].to_numpy()
        pos = np.arange(len(df)) if zone is None else np.flatnonzero(zones == zone)
        horizon = pd.Timedelta(days=cfg.horizon_days)

        est = None
        out = []
        n_eval = 0
        for d in tqdm(np.sort(dates[pos].unique()), desc=f'Prequential {model.name}', unit='giorno'):
            d = pd.Timestamp(d)
            test_pos = pos[dates[pos] == d]
            train_pos = pos[dates[pos] <= d - horizon]
            if len(train_pos) < cfg.min_train_days:
                continue
            if zone is None and not set(zones[test_pos]) <= set(zones[train_pos]):
                continue
            if est is None or n_eval % refit_every == 0:
                est = self._wrap(model, variant)
                est.fit(df[cols].iloc[train_pos], df[Y_COL].iloc[train_pos])
            n_eval += 1
            out.append(pd.DataFrame({
                'giorno': d,
                'zone': zones[test_pos],
                'y_true': df[Y_COL].iloc[test_pos].to_numpy(),
                'y_pred': est.predict(df[cols].iloc[test_pos]),
                'persistance': df[WP_COL].iloc[test_pos].to_numpy(),
            }))

        if not out:
            raise ValueError('No predictable days: reduc min_train_days')
        res = pd.concat(out, ignore_index=True)
        m, b = Evaluator.metrics(res.y_true, res.y_pred), Evaluator.metrics(res.y_true, res.persistance)
        tqdm.write(f'Prequential {model.name} [{variant.label}]: MAE {m['mae']:.3f} vs '
                   f'persistance {b['mae']:.3f} on {len(res)} rows'
                   )
        return res

#   FINAL MODEL ------------------------------------------------------------------------------------

    def fit_final(self, model: BaseModel, df: pd.DataFrame,
                  features: list[str], variant: Variant, zone: int | None = None
                  ) -> tuple[BaseModel, list[str]]:
        cols = self._columns(features, variant)
        d = df if zone is None else df[df[self.cfg.zone_col] == zone]
        est = self._wrap(model, variant)
        est.fit(d[cols], variant)
        tqdm.write(f'Final fit {est.name}: {len(d)} rows, {len(cols)} columns')
        return est, cols

    def save(self, model: BaseModel, columns: list[str],
             variant: Variant, zone: int | None = None, path: str | Path | None = None
             ) -> Path:
        label = 'pooled' if zone is None else f'zone{zone}'
        path = Path(path) if path else self.cfg.models_dir / f'{label}.joblib'
        joblib.dump({
            'model': model,
            'columns': columns,
            'variant': variant,
            'zone': zone,
            'horizon_days': self.cfg.horizon_days,
            'target': self.cfg.target,
        }, path)
        return path

    @staticmethod
    def load(path: str | Path) -> dict:
        return joblib.load(path)