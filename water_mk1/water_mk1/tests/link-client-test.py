from water_mk1.weather_link_client import WeatherLinkClient
from datetime import datetime, timedelta

client = WeatherLinkClient()

stations = client.get_stations()
#print(stations)

data = client.get_current_conditions(90411)
client.to_csv(data, "condizioni_attuali.csv")

end_time   = datetime.now()
start_time = end_time - timedelta(hours=24)

data = client.get_historical_data(90411, start_time, end_time)
client.to_csv(data, "storico_giorno.csv")