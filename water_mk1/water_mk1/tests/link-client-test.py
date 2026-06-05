from water_mk1.weather_link_client import WeatherLinkClient
from datetime import datetime, timedelta
from tqdm import tqdm

FLORAPULSE_EXCLUDE = [864286, 864288, 955142, 955136, 864298]

client = WeatherLinkClient()

end_time   = datetime.now()
start_time = end_time - timedelta(days=2*365)

total_days = (end_time - start_time).days

all_data = {"station_id": 90411, "sensors": [], "generated_at": int(end_time.timestamp())}

current = start_time
with tqdm(total=total_days, unit="day", desc="Downloading", ncols=80) as pbar:
    while current < end_time:
        next_step = min(current + timedelta(hours=24), end_time)
       
        pbar.set_description_str(current.strftime("%Y-%m-%d"), refresh=True)

        try:
            chunk = client.get_historical_data(90411, current, next_step)
            
            # Escludi FloraPulse non porta 4
            chunk["sensors"] = [
                s for s in chunk.get("sensors", [])
                if s.get("lsid") not in FLORAPULSE_EXCLUDE
            ]

            for sensor in chunk.get("sensors", []):
                existing = next(
                    (s for s in all_data["sensors"] if s["lsid"] == sensor["lsid"]),
                    None
                )
                if existing:
                    existing["data"].extend(sensor.get("data", []))
                else:
                    all_data["sensors"].append({
                        "lsid":        sensor["lsid"],
                        "sensor_type": sensor["sensor_type"],
                        "data":        sensor.get("data", []),
                    })
        except Exception as e:
            tqdm.write(f"Error fetching data for {current.strftime('%Y-%m-%d')}: {e}")

        current = next_step
        pbar.update(1)

client.to_csv(all_data, "two_years_history.csv")
