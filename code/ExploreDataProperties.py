import pandas as pd
import os
from pathlib import Path
home = str(Path.home())

emissions = pd.read_csv(f"{home}/geoweaver_demo/tropomi_epa_kvps_NO2_2019_56.csv")
print("==================>")
print("Describe data")
print(emissions.describe())
print("==================>")
print("==================>")
print("Check shape of data")
print(emissions.shape)
print("==================>")
print("==================>")
print("Check available columns")
print(emissions.columns)
print("==================>")
print("==================>")
print("Any NULL values in features?")
print(emissions.isnull().sum())
print("==================>")
