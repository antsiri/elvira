import requests
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
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
    
    def get_historical_data(self, station_id: int | str, start_time: datetime, end_time: datetime) -> dict:
        return self._get(f"historic/{station_id}", {"start-timestamp": int(start_time.timestamp()), "end-timestamp": int(end_time.timestamp())})
    
    def to_csv(self, data: dict, filename: str = "output.csv") -> str:

        station_id = data.get("station_id", "")
        generated_at = data.get("generated_at", "")
        sensors  = data.get("sensors", [])

        output_dir = Path(__file__).resolve().parents[3] / "data"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / filename

        rows = []
        for sensor in sensors:
            lsid = sensor.get("lsid", "")
            sensor_type = sensor.get("sensor_type", "")
            for record in sensor.get("data", []):
                row = {
                    "station_id": station_id,
                    "generated_at": datetime.fromtimestamp(generated_at).isoformat(),
                    "lsid": lsid,
                    "sensor_type": sensor_type,
                } 

                if "ts" in record and record["ts"]:
                    row["ts"] = datetime.fromtimestamp(record["ts"]).isoformat()
                else:
                    row["ts"] = ""

                for key, value in record.items():
                     if key != "ts":
                        row[key] = value

                rows.append(row)

        if not rows:
            print("No data to write to CSV.")
            return ""
        
        all_keys = list(dict.fromkeys(k for row in rows for k in row))
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        print(f"CSV file created at: {output_path}")
        return output_path