import json
import pandas as pd
import ee

try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# identify a 500 meter buffer around our Point Of Interest (POI)
poi = ee.Geometry.Point(-87.910747, 31.488019).buffer(500)

mcd19a2 = ee.ImageCollection("MODIS/006/MCD19A2_GRANULES").filterDate('2019-01-01','2019-03-31')
def poi_mean(img):
    mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=250).get('Optical_Depth_047')
    return img.set('date', img.date().format()).set('mean',mean)

    
# Map function to our ImageCollection
poi_reduced_imgs = tropomiCollection.map(poi_mean)
nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2)['date','mean']).values().get(0)

# we need to call the callback method "getInfo" to retrieve the data
df = pd.DataFrame(nested_list.getInfo(), columns=['date','tropomi_no2_mean'])
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
	# Scaling the data to later match our target feature scale
df['tropomi_no2_mean'] = df['tropomi_no2_mean']*1000
# Save data to CSV file
df.to_csv('tropomi_no2.csv')


