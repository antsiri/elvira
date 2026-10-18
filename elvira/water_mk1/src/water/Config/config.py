"""
 * author Antonio Sirignano
 * created on 12-10-2026-12h-42m
 * github: https://github.com/antsiri
 * copyright 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Config:

    # --- PATHS ---
    data_dir: Path = BASE_DIR / 'data'
    daily_filename: str = 'vigneto_daily_finale.csv'
    hourly_filename: str = 'vigneto_clean_hourly.csv'
    solem_filename: str = 'solem_daily.csv'
    models_dir: Path = BASE_DIR / 'models'

    # --- TARGET ---
    target: str = 'water_potential_last'
    zone_col: str = 'irrigted'                                              #0 = not irrigated, 1 = irrigated
    date_cols: str = 'giorno'
    horizon_days: int = 1                                                   #prediction of wp(t + horizon) with day t features

    # --- LEAKAGE ---
    leakege_cols = ('direct_reading_last', 'watermark_available')

    # --- FEATURES ENGINEERING ---
    lag_windows: tuple = (1, 2, 3, 7)                                        #days, cumulated by liters/ET/rain
    use_autoregressive: bool = False                                        #include wp(t) in the features
    max_missing_hours_per_day: int = 2                                      #threshold, if hour missing are > of the value, day is discarted

    # --- VALIDATION ---
    min_train_days: int = 40
    n_splits: int = 5
    gap_days: int = 1                                                       #>= of horizon_days

    # --- IRRIGATION DECISION ---
    wp_low: float | None = None
    wp_high: float | None = None

    random_state: int = 42

    def __post_init__(self) -> None:
        if self.horizon_days < 1:
            raise ValueError('horizon_days must be >= 1')
        if not self.lag_windows or any(w < 1 for w in self.lag_windows):
            raise ValueError('lag_windows must be contain int >= 1')
        if not 0 <= self.max_missing_hours_per_day <= 23:
            raise ValueError('max_missing_hours_per_day must be between 0 and 23')
        if self.min_train_days < 1:
            raise ValueError('min_train_day must be >= 1')
        if self.n_splits < 2:
            raise ValueError('n_splits must be >= 2')
        if self.gap_days < self.horizon_days:
            raise ValueError('gap_days must be >= horizon_days to avoid leakage')
        if (self.wp_low is None) != (self.wp_high is None):
            raise ValueError('wp_low and wp_high must be setted together')
        if self.wp_low is not None and self.wp_low >= self.wp_high:
            raise ValueError('wp_low must be < wp_high')

    @property
    def daily_path(self) -> Path:
        return self.data_dir / self.daily_filename

    @property
    def hourly_path(self) -> Path:
        return self.data_dir / self.hourly_filename

    @property
    def solem_path(self) -> Path:
        return self.data_dir / self.solem_filename

    @property
    def has_irrigation_range(self) -> bool:
        return self.wp_low is not None