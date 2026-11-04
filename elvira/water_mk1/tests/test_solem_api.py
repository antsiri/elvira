"""
 * author Antonio Sirignano
 * created on 02-11-2026-21h-54m
 * github: https://github.com/antsiri
 * copyright 2026
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from water.Config.config import Config           # noqa: E402
from water.Solem.solem import SolemImporter     # noqa: E402
from water.Ingestion.solem_client import SolemClient, SolemError   # noqa: E402


def make_ticks(start: str, minutes: int, lph: float, counter: float) -> tuple[list[dict], float]:
    """Una corsa: un tick al minuto con litri = lph/60 (intero)."""
    per_min = round(lph / 60)
    ticks = []
    t = pd.Timestamp(start, tz="UTC")
    for i in range(minutes):
        counter += per_min
        ticks.append({"tickTimestamp": (t + pd.Timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:00.000Z"),
                      "value": counter})
    return ticks, counter


class TestSolemApi(unittest.TestCase):
    def setUp(self):
        self.imp = SolemImporter(Config())
        base = [{"tickTimestamp": "2026-06-20T08:00:00.000Z", "value": 1000.0}]       # riferimento
        a, c = make_ticks("2026-06-28T23:11", 60, 1500, 1000.0)                        # 25 L/min x 60
        noise = [{"tickTimestamp": "2026-06-30T10:00:00.000Z", "value": c + 1}]        # goccia isolata
        b, c = make_ticks("2026-07-01T02:11", 30, 1500, c + 1)
        self.ticks = base + a + noise + b
        self.total = c - 1000.0

    def test_litres_match_counter(self):
        daily = self.imp.build_from_ticks(self.ticks)
        self.assertEqual(daily.liters_irrigation.sum(), self.total - 1)       # meno la goccia da 1 L
        self.assertEqual(len(daily), 2)

    def test_first_tick_after_long_gap_is_counted(self):
        # il primo tick arriva dopo giorni di silenzio ma fa parte della corsa
        ses = self.imp.detect_sessions_counter(self.imp.ticks_to_readings(self.ticks))
        self.assertEqual(ses.loc[0, "liters"], 25 * 60)
        self.assertEqual(ses.loc[0, "start"], pd.Timestamp("2026-06-28T23:11", tz="UTC"))

    def test_day_is_utc_start(self):
        daily = self.imp.build_from_ticks(self.ticks)
        self.assertEqual(list(daily.day.dt.strftime("%Y-%m-%d")), ["2026-06-28", "2026-07-01"])

    def test_isolated_drop_is_noise(self):
        ses = self.imp.detect_sessions_counter(self.imp.ticks_to_readings(self.ticks))
        self.assertEqual(len(ses), 2)

    def test_min_session_litres(self):
        ticks = self.ticks + [{"tickTimestamp": "2026-07-02T12:00:00.000Z", "value": self.total + 1000 + 3},
                              {"tickTimestamp": "2026-07-02T12:01:00.000Z", "value": self.total + 1000 + 6}]
        small = self.imp.detect_sessions_counter(self.imp.ticks_to_readings(ticks))
        self.assertEqual(len(small), 3)
        kept = self.imp.detect_sessions_counter(self.imp.ticks_to_readings(ticks), min_session_litres=20)
        self.assertEqual(len(kept), 2)

    def test_counter_reset_is_dropped(self):
        ticks = self.ticks + [{"tickTimestamp": "2026-07-05T00:00:00.000Z", "value": 5.0}]
        r = self.imp.ticks_to_readings(ticks)
        self.assertTrue((r.value >= 0).all())
        
    def test_extract_ticks(self):
        resp = [{"id": "in1", "expression": "x", "computedSensorData": [{"tickTimestamp": "t1", "value": 5}]},
                {"id": "in2", "expression": "x", "computedSensorData": []}]
        self.assertEqual(SolemClient.extract_ticks(resp, "in1"), [{"tickTimestamp": "t1", "value": 5.0}])
        with self.assertRaises(SolemError):
            SolemClient.extract_ticks(resp, "nope")

    def test_empty(self):
        self.assertTrue(self.imp.build_from_ticks([{"tickTimestamp": "2026-06-20T08:00:00.000Z", "value": 1.0}]).empty)


if __name__ == "__main__":
    unittest.main()