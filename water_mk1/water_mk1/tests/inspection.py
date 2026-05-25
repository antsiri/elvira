import pandas as pd

file_path = "/Users/antoniosirignano/Desktop/TIROCINO/water_mk1/data/one_month_history.csv"

df = pd.read_csv(file_path)

print(df.head())

from water_mk1.weather_link_client import WeatherLinkClient
import json

client = WeatherLinkClient()
data = client._get('sensors', {'station-id': 90411})
print(json.dumps(data, indent=2))