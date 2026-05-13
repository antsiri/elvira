import pandas as pd

file_path = "/Users/antoniosirignano/Desktop/TIROCINO/water_mk1/data/one_month_history.csv"

df = pd.read_csv(file_path)

print(df.head())