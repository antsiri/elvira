from water.src.water.Ingestion.weather_link_client import WeatherLinkClient
from datetime import datetime, timedelta
from tqdm import tqdm

FLORAPULSE_EXCLUDE = [864286, 864288, 955136, 864298]  # rimosso 955142

IRRIGATED_MAP = {
    604719: 1,  # IRR2 - TDR port 1
    604720: 1,  # IRR2 - TDR port 2
    604721: 1,  # IRR2 - TDR port 3
    934560: 1,  # IRR2 - FloraPulse porta 4
    604755: 0,  # NOirr2 - TDR port 1
    604759: 0,  # NOirr2 - TDR port 2
    863804: 0,  # NOirr2 - TDR port 4
    955142: 0,  # NOirr2 - FloraPulse porta 3
}

client = WeatherLinkClient()

end_time   = datetime.now()
start_time = end_time - timedelta(days=4*365)

total_days = (end_time - start_time).days

all_data = {"station_id": 90411, "sensors": [], "generated_at": int(end_time.timestamp())}

current = start_time
with tqdm(total=total_days, unit="day", desc="Downloading", ncols=80) as pbar:
    while current < end_time:
        next_step = min(current + timedelta(hours=24), end_time)
       
        pbar.set_description_str(current.strftime("%Y-%m-%d"), refresh=True)

        try:
            chunk = client.get_historical_data(90411, current, next_step)

            # Debug
            # lsids_in_chunk = [s["lsid"] for s in chunk.get("sensors", [])]
            # tqdm.write(f"{current.date()} lsids: {lsids_in_chunk}")

            # chunk["sensors"] = [
            #     s for s in chunk.get("sensors", [])
            #     if s.get("lsid") not in FLORAPULSE_EXCLUDE
            # ]
            # tqdm.write(f"Dopo filtro: {[s['lsid'] for s in chunk['sensors']]}")

            # for sensor in chunk.get("sensors", []):
            #     lsid = sensor["lsid"]
            #     if lsid in IRRIGATED_MAP:
            #         tqdm.write(f"Trovato lsid {lsid} -> irrigated={IRRIGATED_MAP[lsid]}, records={len(sensor.get('data',[]))}")
            #         for record in sensor.get("data", []):
            #             record["irrigated"] = IRRIGATED_MAP[lsid]
            
            chunk["sensors"] = [
                s for s in chunk.get("sensors", [])
                if s.get("lsid") not in FLORAPULSE_EXCLUDE
            ]

            for sensor in chunk.get("sensors", []):
                lsid = sensor["lsid"]
                for record in sensor.get("data", []):
                    record["irrigated"] = IRRIGATED_MAP.get(lsid, None)  # None per meteo

                existing = next(
                    (s for s in all_data["sensors"] if s["lsid"] == lsid),
                    None
                )
                if existing:
                    existing["data"].extend(sensor.get("data", []))
                else:
                    all_data["sensors"].append({
                        "lsid":        lsid,
                        "sensor_type": sensor["sensor_type"],
                        "data":        sensor.get("data", []),
                    })

        except Exception as e:
            tqdm.write(f"Error fetching data for {current.strftime('%Y-%m-%d')}: {e}")

        current = next_step
        pbar.update(1)

client.to_csv(all_data, "history_last_4_years_irr_noirr_28_07.csv")
