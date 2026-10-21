"""
 * author Antonio Sirignano
 * created on 12-10-2026-14h-20m
 * github: https://github.com/antsiri
 * copyright 2026
"""
from __future__ import annotations

from typing import Iterator
from tqdm import tqdm

import numpy as np
import pandas as pd

from water.Config.config import Config

class WalkForwardValidator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def split(self, df: pd.DataFrame, zone: int | None = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        cfg = self.cfg
        dates_all = pd.DatetimeIndex(df[cfg.date_col])
        zones_all = df[cfg.zone_col].to_numpy()

        subset_pos = np.arange(len(df)) if zone is None else np.flatnonzero(zones_all == zone)
        if len(subset_pos) == 0:
            raise ValueError(f'No rows per zone={zone}')
        sub_dates = dates_all[subset_pos]
        sub_zones = zones_all[subset_pos]
        unique_dates = np.sort(sub_dates.unique())

        first_test = cfg.min_train_days + cfg.gap_days
        remaining = len(unique_dates) - first_test
        if remaining < cfg.n_splits:
            raise ValueError(
                f'Not enoigh data: {len(unique_dates)} dates, min_train_days={cfg.min_train_days},'
                f'gap_days={cfg.gap_days} leave {max(remaining, 0)} dates for test per'
                f'n_splits={cfg.n_splits}. Reduce n_splits or min_train_days.'
            )

        size = remaining // cfg.n_splits
        bounds = [first_test + k * size for k in range(cfg.n_splits)] + [len(unique_dates)]

        for k in range(cfg.n_splits):
            test_start = pd.Timestamp(unique_dates[bounds[k]])
            test_end = pd.Timestamp(unique_dates[bounds[k + 1] - 1])
            train_cutoff = test_start - pd.Timedelta(days=cfg.gap_days + 1)

            train_mask = sub_dates <= train_cutoff
            test_mask = (sub_dates >= test_start) & (sub_dates <= test_end)
            if not train_mask.any():
                raise ValueError(f'Fold {k}: empty train (cutoff {train_cutoff.date()})')

            if zone is None:
                missing = set(sub_zones[test_mask]) - set(sub_zones[train_mask])
                if missing:
                    tqdm.write(
                        f'fold {k} skipped: zone {sorted(missing)} in test but not in train'
                        f'(test {test_start.date()} -> {test_end.date()})'
                    )
                    continue

            last_train = sub_dates[train_mask].max()
            assert last_train + pd.Timedelta(days=cfg.horizon_days) < test_start, 'temporal leakage'

            yield subset_pos[train_mask], subset_pos[test_mask]

    def n_folds(self, df: pd.DataFrame, zone: int | None = None) -> int:
        return sum(1 for _ in self.split(df, zone))

    def summary(self, df: pd.DataFrame, zone: int | None = None) -> int:
        cfg = self.cfg
        rows = []

        for k, (tr, te) in enumerate(self.split(df, zone)):
            rows.append({
                'fold': k,
                'n_train': df[cfg.date_col].iloc[tr].min().date(),
                'train_start': df[cfg.data_col].iloc[tr].max().date(),
                'train_end': df[cfg.data_col].iloc[tr].min().date(),
                'test_start': df[cfg.data_col].iloc[te].min().date(),
                'train_start': df[cfg.data_col].iloc[te].max().date(),
            })

        return pd.DataFrame(rows)
