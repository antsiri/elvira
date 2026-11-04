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

    def coverage_gaps(self, readings: pd.DataFrame, min_days: int = 50) -> pd.DataFrame:
        gap = readings['ts'].diff()
        long = gap[gap > pd.Timedelta(days=min_days)]
        out = pd.DataFrame({
            'da': readings['ts'].shift(1).loc[long.index].dt.date,
            'a': readings['ts'].loc[long.index].dt.date,
            'days': long.dt.days,
        }).reset_index(drop=True)

        if len(out):
            tqdm.write(f'{len(out)} holes > {min_days} fays betwwen lectures')
        return out

#   SESSIONS  --------------------------------------------------------------------------------------

    def detect_sessions(self, readings: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        r = readings.copy()
        r['gap_h'] = r['ts'].diff().dt.total_seconds() / 3600
        r['rate'] = r['value'] / r['gap_h']
        flow = r[r['rate'] > cfg.solem_flow_threshold_lhp].copy()
        if flow.empty:
            return pd.DataFrame(columns=['start', 'end', 'liters', 'n_lectures', 'min_duration'])

        inner = flow['ts'].diff().dt.total_seconds() / 3600
        flow['session_id'] = ((inner > cfg.solem_session_gap_h) | inner.isna()).cumsum()

        ses = flow.groupby('session_id').agg(
            start=('ts', 'min'),
            end=('ts', 'max'),
            liters=('value', 'sum'),
            n_lectures=('value', 'count'),
            gap_before=('gap_h', 'first')
        ).reset_index(drop=True)
        ses['min_duration'] = (ses['end'] - ses['start']).dt.total_seconds() / 60
        single = ses['n_lectures'] == 1
        ses.loc[single, 'min_duration'] = ses.loc[single, 'gap_before']
        return ses.drop(columns='gap_before')

    def to_daily(self, sessions: pd.DataFrame) -> pd.DataFrame:
        if sessions.empty:
            return pd.DataFrame(columns=['day', 'liters_irrigation', 'n_sessions', 'min_duration'])
        s = sessions.assign(day=sessions['start'].dt.tz_convert('UTC').dt.tz_localize(None).dt.normalize())
        return (s.groupby('day').agg(liters_irrigation=('liters', 'sum'),
                                     n_sessions=('start', 'count'),
                                     min_duration=('min_duration', 'sum'))
                                     .reset_index())

    def start_hour_report(self, sessions: pd.DataFrame) -> pd.DataFrame:
        h_utc = sessions['start'].dt.tz_convert('UTC').dt.hour
        h_loc = sessions['start'].dt.tz_convert('Europe/Rome').dt.hour
        return pd.DataFrame({
            'hour': range(24),
            'utc_sessions': h_utc.value_counts().reindex(range(24), fill_value=0).to_numpy(),
            'local_sessions': h_loc.value_counts().reindex(range(24), fill_value=0).to_numpy(),
        })

#   PIPELINE  --------------------------------------------------------------------------------------

    def build(self, paths: list[Path] | None = None, save: bool = False) -> pd.DataFrame:
        readings = self.load_readings(paths)
        self.coverage_gaps(readings)
        sessions = self.detect_sessions(readings)
        daily = self.to_daily(sessions)
        tqdm.write(f'Sessions: {len(sessions)}, irrigation days: {len(daily)}, total liters: {daily.irrifation_liters.sum():0f}')
        if save:
            daily.to_csv(self.cfg.solem_path, index=False)
            tqdm.write(f'Saved: {self.cfg.solem_path}')
        return daily

#   API MySolem  -----------------------------------------------------------------------------------

    @staticmethod
    def ticks_to_readings(ticks: list[dict]) -> pd.DataFrame:
        if len(ticks) < 2:
            return pd.DataFrame(columns=['ts', 'value', 'gap_min'])
        df = pd.DataFrame(ticks)
        df['ts'] = pd.to_datetime(df['tickTimestamp'], utc=True)
        df = df.drop_duplicates('ts', keep='last').sort_values('ts').reset_index(drop=True)
        df['liters'] = df['value'].diff()
        df['gap_min'] = df['ts'].diff().dt.total_seconds() / 60
        df = df.iloc[1:]
        neg = df['liters'] < 0
        if neg.any():
            tqdm.write(f'Counter: {int(neg.sum())} negative differences discarted (zeroing?)')
            df = df[~neg]
        return (df[['ts', 'liters', 'gap_min']].rename(columns={'liters': 'value'}).reset_index(drop=True))

    def detect_sessions_counter(self, readings: pd.DataFrame, min_session_litres: float = 0.0) -> pd.DataFrame:
        cfg = self.cfg
        cols = ['start', 'end', 'liters', 'n_lectures', 'min_duration']
        if readings.empty:
            return pd.DataFrame(columns=cols)
        r = readings.copy()
        eff = pd.concat({'before': r['gap_min'], 'after': r['gap_min'].shift(-1)}, axis=1).min(axis=1).clip(lower=1.0)
        r['rate'] = r['value'] * 60 / eff
        flow = r[r['rate'] > cfg.solem_flow_threshold_lhp].copy()
        if flow.empty:
            return pd.DataFrame(columns=cols)
        inner = flow['ts'].diff().dt.total_seconds() / 3600
        flow['session_id'] = ((inner > cfg.solem_session_gap_h) | inner.isna()).cumsum()
        ses = flow.groupby('session_id').agg(
            start=('ts', 'min'),
            end=('ts', 'max'),
            liters=('value', 'sum'),
            n_lectures=('value', 'count')
        ).reset_index(drop=True)
        ses['min_duration'] = (ses['end'] - ses['start']).dt.total_seconds() / 60 + 1.0
        small = ses['liters'] < min_session_litres
        if small.any():
            tqdm.write(f'Discarded {int(small.sum())} micro-sessions under {min_session_litres:0f} L')
            ses = ses[~small].reset_index(drop=True)
        return ses

    def build_from_ticks(self, ticks: list[dict], min_session_liters: float = 0.0) -> pd.DataFrame:
        readings = self.ticks_to_readings(ticks)
        sessions = self.detect_sessions_counter(readings, min_session_liters)
        daily = self.to_daily(sessions)
        tqdm.write(f'tick: {len(ticks)}, sessions: {len(sessions)}, irrigated days: {len(daily)}, total liters: {daily.liters_irrigation.sum():.0f}')
        return daily

