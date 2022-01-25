import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
home = str(Path.home())

plt.style.use('fivethirtyeight')
emissions = pd.read_csv(f"{home}/geoweaver_demo/tropomi_epa_kvps_NO2_2019_56.csv")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,8))  # 1 row, 2 columns

emissions.plot(x='EPA_NO2/100000', y='TROPOMI*1000', kind='scatter', color='orange', ax=ax1)

emissions.plot(x='Wind (Monthly)', y='Cloud Fraction (Monthly)', kind='scatter', color='green', ax=ax2)

ax1.set_xlabel('EPA_NO2/100000',fontsize=15)
ax2.set_xlabel('Wind (Monthly)',fontsize=15)
ax1.set_ylabel('TROPOMI*1000',fontsize=15)
ax2.set_ylabel('Cloud Fraction (Monthly)',fontsize=15)

emissions.plot(x='Temp (Monthly)', y='EPA_NO2/100000', kind='scatter', color='blue', ax=ax3)

emissions.plot(x='EPA_NO2/100000', y='Cloud Fraction (Monthly)', kind='scatter', color='red', ax=ax4)

ax3.set_xlabel('Temp (Monthly)',fontsize=15)
ax4.set_xlabel('EPA_NO2/100000',fontsize=15)
ax3.set_ylabel('TROPOMI*1000',fontsize=15)
ax4.set_ylabel('Cloud Fraction (Monthly)',fontsize=15)

plt.savefig(f'{home}/geoweaver_demo/features.png')

