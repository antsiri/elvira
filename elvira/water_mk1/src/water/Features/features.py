"""
 * author Antonio Sirignano
 * created on 15-10-2026-19h-52m
 * github: https://github.com/antsiri
 * copyright 2026
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from tqdm import tqdm

from water.Config.config import Config

LITRI_COL = 'litri_irrigati'

WEATHER_BASE = (
    'et_mm', 'rainfall_mm', 'deg_days_cool',
    'deg_days_heat', 'rain_rate_hi_mm', 'wind_speed_hi_kmh',
    'hum_out', 'bar_hpa', 'thsw_index_c',
    'night_cloud_cover', 'bar_trend_ord'
)

SOIL_PROBE = ('moist_soil_last_1', 'salinity_last_1', 'temp_last_1_c')

WIND_DIR = ('wind_dir_of_hi', 'wind_dir_of_prevail')
class FeaturesBuilder:
    baseline_col = 'wp_t'
    target_col = 'y'

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.features_names_: list[str] = []

    def build(self, daily: pd.DataFrame) -> pd.DataFrame:
        self.features_names_ = [self.cfg.zone_col]
        with tqdm(total=6, desc='Feature engineering', unit='step') as bar:
            df = self._complete_calendar(daily); bar.update()
            df = self._add_base_features(df); bar.update()
            df = self._add_history_features(df); bar.update()
            df = self._add_water_balance(df); bar.update()
            df = self._add_target(df); bar.update()
            out = self._finalize(df); bar.update()

        self._check_no_leakage()
        return out

#   CALENDAR  --------------------------------------------------------------------------------------

    def _complete_calendar(self, daily: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        parts = []
        for zone, g in daily.groupby(cfg.zone_col):
            full = pd.date_range(g[cfg.date_col].min())
            g = g.set_index(cfg.date_col).reindex(full).rename_axis(cfg.date_col).reset_index()
            g[cfg.date_col] = int(zone)
            parts.append(g)
        out = pd.concat(parts, ignore_index=True)

        out.loc[out[cfg.zone_col] == 0, LITRI_COL] = 0.0
        out = out.sort_values([cfg.zone_col, cfg.date_col]).reset_index(drop=True)

        n_new = len(out) - len(daily)
        if n_new:
            tqdm.write(f'Calendar: missing day {n_new} inserted (NaN rows)')
        return out

    def _add_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in WIND_DIR:
            angle = df[col] * (2 * np.pi / 16)
            df[f'{col}_sin'] = np.sin(angle)
            df[f'{col}_cos'] = np.cos(angle)

        wind_cols = [f'{c}_{f}' for c in WIND_DIR for f in ('sin', 'cos')]

        df['litri_t'] = df[LITRI_COL]
        self.features_names_ += list(WEATHER_BASE) + list(SOIL_PROBE) + wind_cols + ['litri_t']
        return df

#   HISTORY  ---------------------------------------------------------------------------------------

    def _rulling_sum(self, df: pd.DataFrame, col: str, window: int, lag: int) -> pd.Series:
        return df.groupby(self.cfg.zone_col)[col].transform(
            lambda s: s.shift(lag).rolling(window,  min_periods=window).sum()
        )
        
    def _add_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def _add_water_balance(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for w in self.cfg.lag_windows:
            if w > 1:
                df[f'balance_{w}d'] = df[f'rain_sum_{w}d'] - df[f'et_sum_{w}d']
                self.features_names_.append(f'balance_{w}d')
        return df

#   TARGET  ----------------------------------------------------------------------------------------

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        df = df.copy()
        df[self.baseline_col] = df[cfg.target]
        df[self.target_col] = df.groupby(cfg.zone_col)[cfg.target].shift(-cfg.horizon_days)
        if cfg.use_autoregressive:
            self.features_names_.append(self.baseline_col)
        return df

#   CLEANING  --------------------------------------------------------------------------------------

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        required = self.features_names_ + [self.baseline_col, self.target_col]
        candidates = df[df[self.target_col].notna()]
        out = candidates.dropna(subset=required)

        lost = len(candidates) - len(out)
        if lost:
            nan_cols = candidates[required].isna().sum()
            nan_cols = nan_cols[nan_cols > 0].sort_values(ascending=False)
            tqdm.write(
                f'rows with target for NaN in their features: {lost}'
                f'(columns involved {nan_cols.to_dict()})'
            )
        per_zone = out.groupby(cfg.zone_col).size().to_dict()
        tqdm.write(f'Final Dataset: {len(out)} rows, {len(self.features_names_)} features, per zone {per_zone}')

        cols = list(dict.fromkeys(
            [cfg.date_col, cfg.zone_col] + self.features_names_ + [self.baseline_col, self.target_col]
        ))

        return out[cols].sort_values([cfg.date_col, cfg.zone_col]).reset_index(drop=True)


    def split_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if not self.features_names_:
            raise RuntimeError('Use build() before split_xy()')
        return df[self.features_names_].copy(), df[self.target_col].copy()

    def _check_no_leakage(self) -> None:
        banned = set(self.cfg.leakege_cols) | {self.cfg.target, self.target_col, self.cfg.date_col}
        found = banned & set(self.features_names_)
        if found:
            raise ValueError(f'Not allowed columns in features: {sorted(found)}')
        