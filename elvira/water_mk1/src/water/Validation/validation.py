"""
 * author Antonio Sirignano
 * created on 12-10-2026-14h-20m
 * github: https://github.com/antsiri
 * copyright 2026
"""

from typing import Iterator

import numpy as np
import pandas as pd

from water.Config.config import Config

class WalkForwardValidator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def split(self, df: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:

        raise NotImplementedError