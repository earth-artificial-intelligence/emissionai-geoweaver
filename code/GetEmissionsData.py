import pandas as pd
import os
from pathlib import Path
home = str(Path.home())
emissions = pd.read_csv('https://raw.githubusercontent.com/ZihengSun/EmissionAI/main/data/tropomi_epa_kvps_NO2_2019_56.csv' , parse_dates=["Date"])
print(emissions)
demo_dir = f"{home}/geoweaver_demo/"
if not os.path.exists(demo_dir):
	os.mkdir(demo_dir)
emissions.to_csv(f"{home}/geoweaver_demo/tropomi_epa_kvps_NO2_2019_56.csv")
