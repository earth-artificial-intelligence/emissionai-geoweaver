from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import metrics 
from sklearn import tree
import os
from pathlib import Path
home = str(Path.home())

def showAccuracyMetrics(mlmethod, model, y_test, y_pred):    
	print("Model ", mlmethod, " Performance:")
	#print(y_test.shape, y_pred.shape)	
	mae = metrics.mean_absolute_error(y_test, 			y_pred)
	mse = metrics.mean_squared_error(y_test, 			y_pred)
	r2 = metrics.r2_score(y_test, y_pred)
	print("   MAE: ", mae)
	print("   MSE: ", mse)    
	print("   R2: ", r2)

regression_emissions = pd.read_csv(f'{home}/geoweaver_demo/preprocessed.csv')
    
target_column = ['EPA_NO2/100000'] 
predictors = ['TROPOMI*1000', 'dayofyear', 'dayofweek', 'dayofmonth']

all_X = regression_emissions[predictors]
all_y = regression_emissions[target_column]


xtrain, xtest, ytrain, ytest = train_test_split(all_X,all_y,test_size=0.30, random_state=42)


randomForestregModel = RandomForestRegressor(max_depth=15)
randomForestregModel.fit(xtrain, np.ravel(ytrain))

ypred = randomForestregModel.predict(xtest)


showAccuracyMetrics("RF: ", randomForestregModel, ytest, ypred)



fn=all_X.columns
cn=all_y.columns
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (0.5,0.5), dpi=800)
out = tree.plot_tree(randomForestregModel.estimators_[0],
               feature_names = fn, 
#                class_names=cn,
               filled = True,
               );

for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor('black')
        arrow.set_linewidth(.1)
        
plt.savefig(f'{home}/geoweaver_demo/tree.eps',format='eps',bbox_inches = "tight")
