import requests
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import csv
import os

load_dotenv()

class WeatherLinkClient:

    BASE_URL = "https://api.weatherlink.com/v2/"

    def __init__(self):
        self.api_key = os.getenv("WEATHERLINK_API_KEY")
        self.api_key_secret = os.getenv("WEATHERLINK_API_SECRET")

        # print(f"API_KEY loaded: {bool(self.api_key)}")
        # print(f"API_SECRET loaded: {bool(self.api_key_secret)}")

        if not self.api_key or not self.api_key_secret:
            raise ValueError("API key and secret must be set in environment variables")
        
        self.headers = {"X-Api-Secret": self.api_key_secret}

    def _get(self, path: str, extra_params: dict = None) -> dict:
        params = {"api-key": self.api_key}
        if extra_params:
            params.update(extra_params)

        response = requests.get(f"{self.BASE_URL}{path}", headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def get_stations(self) -> dict:
        return self._get("stations")
    
    def get_current_conditions(self, station_id: int | str) -> dict:
        return self._get(f"current/{station_id}")
    
    def get_historical_data(self, station_id: int | str, start_time: datetime, end_time: datetime, lsid_filter: list = None) -> dict:
        data =  self._get(f"historic/{station_id}", {"start-timestamp": int(start_time.timestamp()), "end-timestamp": int(end_time.timestamp())})

        if lsid_filter:
            data["sensors"] = [
                s for s in data.get("sensors", []) if s.get("lsid") in lsid_filter
            ]

        return data
    
    def to_csv(self, data: dict, filename: str = "output.csv") -> str:

        KEEP_COLUMNS = {
            "ts",
            "temp_out", "temp_out_hi", "temp_out_lo",
            "hum_out", "dew_point_out", "wet_bulb",
            "wind_speed_avg", "wind_speed_hi", "wind_dir_of_hi",
            "wind_dir_of_prevail", "wind_run", "wind_chill",
            "solar_rad_avg", "solar_rad_hi", "solar_energy",
            "uv_index_avg", "uv_index_hi", "uv_dose",
            "bar", "bar_alt", "abs_press", "bar_trend_3_hr", "pressure_last",
            "rainfall_in", "rainfall_mm", "rain_rate_hi_in", "rain_rate_hi_mm",
            "et", "heat_index_out", "thw_index", "thsw_index",
            "deg_days_heat", "deg_days_cool", "air_density", "emc",
            "night_cloud_cover",
            "moist_soil_last_1", "temp_last_1", "salinity_last_1",
            "water_potential_last", "direct_reading_last", "temp_resistance_last",
            "irrigated"
        }

        station_id = data.get("station_id", "")
        generated_at = data.get("generated_at", "")
        sensors  = data.get("sensors", [])

        output_dir = Path(__file__).resolve().parents[3] / "data"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / filename

        frames = []
        for sensor in sensors:
            lsid = sensor.get("lsid", "")
            sensor_type = sensor.get("sensor_type", "")
            records = sensor.get("data", [])

            if not records:
                print(f"Warning: No data records for sensor {lsid} ({sensor_type})")
                continue

            df = pd.DataFrame(records)
            if "ts" not in df.columns:
                print(f"Warning: No timestamp column for sensor {lsid} ({sensor_type})")
                continue

            df["ts"] = pd.to_datetime(df["ts"], unit="s").dt.floor("h")

            cols_to_keep = [col for col in df.columns if col in KEEP_COLUMNS]
            if len(cols_to_keep) <= 1:
                continue
            df = df[cols_to_keep]
            frames.append(df)

        if not frames:
            print("No valid sensor data to write.")
            return ""

        combined = pd.concat(frames, ignore_index=True)
        combined["irrigated"] = pd.to_numeric(combined["irrigated"], errors="coerce")

        numeric_cols = combined.select_dtypes(include="number").columns.tolist()
        numeric_cols_no_irr = [c for c in numeric_cols if c != "irrigated"]

        soil_df  = combined[combined["irrigated"].notna()].copy()
        meteo_df = combined[combined["irrigated"].isna()].copy()

        # File 1: meteo aggregato per ts
        meteo_only_cols = [c for c in numeric_cols_no_irr if meteo_df[c].notna().any()]
        meteo_agg = meteo_df.groupby("ts")[meteo_only_cols].mean().reset_index()

        # File 2: suolo aggregato per (ts, irrigated)
        soil_agg = soil_df.groupby(["ts", "irrigated"])[numeric_cols_no_irr].mean().reset_index()

        # Combina: merge su ts
        meteo_agg["ts"] = meteo_agg["ts"].astype(str)
        soil_agg["ts"]  = soil_agg["ts"].astype(str)

        result = pd.merge(soil_agg, meteo_agg, on="ts", how="left", suffixes=("_soil", ""))
        result = result[[c for c in result.columns if not c.endswith("_soil")]]

        result = result.sort_values(["ts", "irrigated"]).reset_index(drop=True)
        result.insert(0, "station_id", station_id)
        result.to_csv(output_path, index=False)
        print(f"Data written to {output_path}")
        return str(output_path)