"""
 * author Antonio Sirignano
 * created on 28-10-2026-21h-39m
 * github: https://github.com/antsiri
 * copyright 2026
"""

from __future__ import annotations

import pandas as pd

from pathlib import Path
from tqdm import tqdm

from water.Config.config import Config

REQUIRED_COLS = ('timestamp', 'value')
FLOW_INPUT_TYPE = 'Flow Sensor'

class SolemImporter:
    def __init__(self, cfg: Config):
        self.cfg = cfg

#   READ  ------------------------------------------------------------------------------------------

    def find_exports(self) -> list[Path]:
        paths = sorted(self.cfg.solem_export_dir.glob(self.cfg.solem_export_glob))
        if not paths:
            raise FileNotFoundError(f'No Solem export in {self.cfg.solem_export_dir} pattern "{self.cfg.solem_export_glob}"')

        return paths

    def load_readings(self, paths: list[Path] | None = None) -> pd.DataFrame:
        paths = paths or self.find_exports()
        frames = []
        for path in tqdm(paths, desc='Reading Solem exports', unit='file'):
            df = pd.read_csv(path)
            missing = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing:
                raise ValueError(f'{path.name}: missing columns {missing}')
            if 'inputType' in df.columns:
                other = df['inputType'].ne(FLOW_INPUT_TYPE)
                if other.any():
                    tqdm.write(f'{path.name}: discarded {int(other.sum())} rows not "{FLOW_INPUT_TYPE}"')
                    df = df[-other]
            frames.append(df[list(REQUIRED_COLS)])

        raw = pd.concat(frames, ignore_index=True)
        raw['ts'] = pd.to_datetime(raw['timestamp'], utc=True)
        n_dup = int(raw['ts'].duplicated().sum())
        readings = (raw[['ts', 'value']].drop_duplicates('ts', keep='last').sort_values('ts').reset_index(drop=True))
        tqdm.write(f'Readings: {len(readings)} from {len(paths)} files {n_dup} duplicates dropped')

        return readings