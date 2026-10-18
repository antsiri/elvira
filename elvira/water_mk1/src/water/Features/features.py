"""
 * author Antonio Sirignano
 * created on 15-10-2026-19h-52m
 * github: https://github.com/antsiri
 * copyright 2026
"""

import pandas as pd

from water.Config.config import Config

class FeaturesBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.features_names_: list[str] = []

    def build(self, daily: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def _complete_calendar(self, daily: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def _add_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def split_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        raise NotImplementedError