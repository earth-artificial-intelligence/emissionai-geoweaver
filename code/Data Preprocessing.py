import pandas as pd
import os
from pathlib import Path
home = str(Path.home())


regression_emissions = pd.read_csv(f"{home}/geoweaver_demo/tropomi_epa_kvps_NO2_2019_56.csv",parse_dates=["Date"])

regression_emissions['dayofyear'] = regression_emissions['Date'].dt.dayofyear
regression_emissions['dayofweek'] = regression_emissions['Date'].dt.dayofweek
regression_emissions['dayofmonth'] = regression_emissions['Date'].dt.day
regression_emissions = regression_emissions.drop(columns=["Date"])


regression_emissions.to_csv(f'{home}/geoweaver_demo/preprocessed.csv')
