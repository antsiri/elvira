"""
 * author Antonio Sirignano
 * created on 18-10-2026-11h-33m
 * github: https://github.com/antsiri
 * copyright 2026
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

from water.Config.config import Config
from water.Models.models import BaseModel
from water.Trainer.trainer import Trainer

LITERS_COL = 'litri_t'

@dataclass
class Recommendation:
    status: str
    liters: float | None
    wp_no_irrigation: float
    wp_expected: float | None
    threshold: float
    note: str = ''

class IrrigationAdvisor:
    def __init__(self, cfg: Config, model: BaseModel, 
                 columns: list[str], volume_grid: np.ndarray, margin: float = 0.0,
                 min_effect: float = 0.05):
        if LITERS_COL not in columns:
            raise ValueError(f'The model doesn\'t use "{LITERS_COL}": impossible to predict the volume.')
        self.cfg = cfg
        self.model = model
        self.columns = list(columns)

        grid = np.unique(np.r_[0.0, np.asarray(volume_grid, dtype=float)])
        if (grid < 0).any():
            raise ValueError('volume_grid contains negative liters')
        
        self.volume_grid = grid
        self.margin = float(margin)
        self.min_effect = float(min_effect)

#   CONSTRUCTION -----------------------------------------------------------------------------------

    @staticmethod
    def grid_from_history(df: pd.DataFrame, zone: int | None, n: int = 25,
                          zone_col: str = 'irrigated') -> np.ndarray:
        d = df if zone is None else df[df[zone_col] == zone]
        pos = d.loc[d[LITERS_COL] > 0, LITERS_COL].to_numpy()
        if len(pos) == 0:
            raise ValueError('No irrigation in history: grid is impossible')
        return np.unique(np.r_[0.0, np.quantile(pos, np.linespace(0.1, 1.0, n-1))])

    @classmethod
    def from_bundle(cls, cfg: Config, path: str | Path, df: pd.DataFrame, **kwarg) -> 'IrrigationAdvisor':
        b = Trainer.load(path)
        grid = cls.grid_from_history(df, b['zone'], zone_col=cfg.zone_col)
        return cls(cfg, b['model'], b['columns'], grid, **kwargs)
    
#   PREDICTION -------------------------------------------------------------------------------------

    def _predict_many(self, state: pd.Series, liters: np.ndarray) -> np.ndarray:
        missing = [c for c in self.columns if c not in state.index]
        if missing:
            raise KeyError(f'Incompleted State, missing: {missing}')

        base = state[self.columns].to_numpy(dtype=float)
        X = pd.DataFrame(np.tile(base, (len(liters), 1)), columns=self.columns)
        X[LITERS_COL] = liters
        return self.model.predict(X)

    def predict_for_volume(self, state: pd.Series, liters: float) -> float:
        return float(self._predict_many(state, np.array([liters]))[0])

    def curve(self, state: pd.Series) -> pd.Series:
        return pd.DataFrame({
            'liters': self.volume_grid,
            'wp_pred': self._predict_many(state, self.volume_grid)
        })
        
#   DECISION ---------------------------------------------------------------------------------------

    def recommend(self, state: pd.Series) -> Recommendation:
        cfg = self.cfg
        if not cfg.has_irrigation_range:
            raise ValueError('Set wp_low and wp_high in Config before call recommend()')
        c = self.curve(state)
        threshold = cfg.wp_low + self.margin
        base = float(c.wp_pred.iloc[0])

        if base >= threshold:
            return Recommendation(status='no_irrigation',
                                  liters=0.0,
                                  wp_no_irrigation=base,
                                  wp_expected=base,
                                  threshold=threshold,
                                  note='wp is already up to threshold without irrigation'
                                  )

        effect = float(c.wp_pred.max() - c.wp_pred.min())
        if effect < self.min_effect:
            return Recommendation(status='insensitive',
                                  liters=None,
                                  wp_no_irrigation=base,
                                  wp_expected=None,
                                  threshold=threshold,
                                  note=f'the model varies only of {effect:.3f} on liters grid: volume non predictible, choose you manually or with a rule'
                                  )

        notes = []
        if (np.diff(c.wp_pred.to_numpy()) < -1e-9).any():
            notes.append('not monotonic curve: more water doesn\'t port at a wp more high')
        ok = c[c.wp_pred >= threshold]
        if ok.empty:
            return Recommendation(status='insufficient',
                                  liters=float(c.liters.iloc[-1]),
                                  wp_no_irrigation=base,
                                  wp_expected=float(c.wp_pred.iloc[-1]),
                                  threshold=threshold,
                                  note='; '.join(['No volumne observed that reach the threshold'] + notes)
                                  )

        row = ok.iloc[0]
        if row.wp_pred > cfg.wp_high:
            notes.append(f'Exceed wp_high ({cfg.wp_high}): the minimum volum pushes the limit')
        return Recommendation(status='irrigate',
                              liters=float(row.liters),
                              wp_no_irrigation=base,
                              wp_expected=float(row.wp_pred),
                              threshold=threshold,
                              note='; '.join(notes)
                              )

#   DIAGNOSTIC -------------------------------------------------------------------------------------

    def check_monotonocity(self, df: pd.DataFrame, n_states: int = 30) -> pd.DataFrame:
        idx = np.unique(np.linspace(0, len(df) - 1, min(n_states, len(df))).astype(int))
        rows = []
        for i in tqdm(idx, desc='Check response in liters', unit='state'):
            c = self.curve(df.iloc[i])
            rows.append({
                'giorno': df[self.cfg.date_col].iloc[i],
                'effect': float(c.wp_pred.max() - c.wp_pred.min()),
                'monotone': bool((np.diff(c.wp_pred.to_numpy()) >= -1e-9).all()),
                'slope_for_1000L': float(np.polyfit(c.liters, c.wp_pred, 1)[0] * 1000),
            })

        out = pd.DataFrame(rows)
        tqdm.write(f'Liter-sensitive (effect >= {self.min_effext}): {(out.effect >= self.min_effect).mean():.0%} of states; '
                f'monotonous: {out.monotone.mean():.0%}: mean slope {out.slope_for_1000L.mean():+3f} wp/1000 L')

        return out

    @staticmethod
    def threshold_report(predictions: pd.DataFrame, wp_low: float) -> pd.DataFrame:
        rows = []
        for (z, v, m), g in predictions.groupby(['zone', 'variant', 'model'], sort=False):
            need, pred = g.y_true < wp_low, g.y_pred < wp_low
            tp, fp = int((need & pred).sum()), int((~need & pred).sum())
            fn, tn = int((need & ~pred).sum()), int((~need & ~pred).sum())
            rows.append({
                'zone': z,
                'varaint': v,
                'model': m,
                'n': len(g),
                'real_positive': int(need.sum()),
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'TN': tn,
                'precision': tp / (tp + fp) if tp + fp else np.nan,
                'recall': tp / (tp + fn) if tp + fn else np.nan,
                'accuracy': (tp + tn) / len(g),
            })
        return pd.DataFrame(rows)
