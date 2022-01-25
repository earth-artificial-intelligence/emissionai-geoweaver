import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.svm import SVR
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

X = regression_emissions[['dayofyear']]
y = regression_emissions['EPA_NO2/100000']
xtrain, xtest, ytrain, ytest = train_test_split(X,y,test_size=0.30, random_state=42)

xtrain = X.iloc[:116]
ytrain = y.iloc[:116]
xtest = X.iloc[116:]
ytest = y.iloc[116:]

svr_rbf = SVR(kernel='rbf', C=1e4, gamma=0.35)
svr_rbf.fit(X, y)
y_rbf = svr_rbf.predict(X)

showAccuracyMetrics("SVR: ", svr_rbf, y, y_rbf)

plt.scatter(X, y, c='k', label='data')
plt.plot(X, y_rbf, c='g', label='RBF model')
plt.xlabel('dayofyear')
plt.ylabel('EPA_NO2/100000')
plt.title('Support Vector Regression')
plt.legend()
plt.savefig(f'{home}/geoweaver_demo/SVR_model.png')
