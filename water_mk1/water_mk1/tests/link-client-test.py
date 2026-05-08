from water_mk1.weather_link_client import WeatherLinkClient
from datetime import datetime, timedelta

client = WeatherLinkClient()

end_time   = datetime.now()
start_time = end_time - timedelta(days=365 * 2)

all_data = {"station_id": 90411, "sensors": [], "generated_at": int(end_time.timestamp())}

current = start_time
while current < end_time:
    next_step = min(current + timedelta(hours=24), end_time)
    print(f"Scaricando {current.date()}...")

    try:
        chunk = client.get_historical_data(90411, current, next_step)
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
        print(f"Errore per {current.date()}: {e}")

    current = next_step

client.to_csv(all_data, "two_years_history.csv")