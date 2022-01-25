import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf
import math


def showAccuracyMetrics(mlmethod, model, y_test, y_pred):    
	print("Model ", mlmethod, " Performance:")
	#print(y_test.shape, y_pred.shape)	
	mae = metrics.mean_absolute_error(y_test, 			y_pred)
	mse = metrics.mean_squared_error(y_test, 			y_pred)
	r2 = metrics.r2_score(y_test, y_pred)
	print("   MAE: ", mae)
	print("   MSE: ", mse)    
	print("   R2: ", r2)

plt.style.use('fivethirtyeight')

emissions_alabama_all = pd.read_csv('~/geoweaver_demo/tropomi_epa_kvps_NO2_2019_56.csv',parse_dates=["Date"])

emissions_alabama_all['dayofyear'] = emissions_alabama_all['Date'].dt.dayofyear
emissions_alabama_all['dayofweek'] = emissions_alabama_all['Date'].dt.dayofweek
emissions_alabama_all['dayofmonth'] = emissions_alabama_all['Date'].dt.day
emissions_alabama_all = emissions_alabama_all.drop(columns=["Date"])

def create_dataset(dataset, look_back=7):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1): 
        a = dataset[i:(i+look_back), 1:] 
#         print(a)
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0]) 
    return np.array(dataX), np.array(dataY)
    
dataset = emissions_alabama_all.values
dataset = dataset.astype('float32')

# normalize the dataset
look_back = 7
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


print("X_train's shape: ", trainX.shape)
print("y_train's shape: ", trainY.shape)
print("x_test's shape: ", testX.shape)
print("y_test's shape: ", testY.shape)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 12, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 12, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adadelta', metrics=[tf.keras.metrics.mean_squared_error, 'accuracy'])
history = model.fit(trainX, trainY, epochs=50, batch_size=1)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

showAccuracyMetrics("LSTM [Alabama Plant All features]: ", model, testY, testPredict)

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
#plt.plot(dataset[:,0])
#plt.plot(trainPredictPlot[:,0])
#plt.plot(testPredictPlot[:,0])
#plt.legend(["Data", "Train", "Test"])
#plt.title("One plant (ID 56, Alabama)")
#plt.savefig('/Users/uhhmed/geoweaver_demo/LSTM_model.png')

