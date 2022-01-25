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
import os

homedir = os.path.expanduser('~')


plt.style.use('fivethirtyeight')
emissions = pd.read_csv(f"{homedir}/geoweaver_demo/tropomi_epa_kvps_NO2_2019_56.csv",parse_dates=["Date"])

def showAccuracyMetrics(mlmethod, model, y_test, y_pred):    
	print("Model ", mlmethod, " Performance:")
	#print(y_test.shape, y_pred.shape)	
	mae = metrics.mean_absolute_error(y_test, 			y_pred)
	mse = metrics.mean_squared_error(y_test, 			y_pred)
	r2 = metrics.r2_score(y_test, y_pred)
	print("   MAE: ", mae)
	print("   MSE: ", mse)    
	print("   R2: ", r2)
    
emissions['dayofyear'] = emissions['Date'].dt.dayofyear
emissions['dayofweek'] = emissions['Date'].dt.dayofweek
emissions['dayofmonth'] = emissions['Date'].dt.day
emissions = emissions.drop(columns=["Date"])

# Separating dependednt & Indepented Variables 
x = emissions.iloc[:, emissions.columns != 'EPA_NO2/100000'].values
y = emissions.iloc[:,  emissions.columns == 'EPA_NO2/100000']

# show the shape of x and y to make sure they have the same length

# Train Test Split at ratio 0.33
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
y_train = y_train.ravel()
y_test = y_test.ravel()

# Define Keras NN model
model = Sequential()
model.add(Dense(500, input_dim=12, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer="adadelta", loss="mse",  metrics=[tf.keras.metrics.mean_squared_error, 'accuracy'])
history = model.fit(x_train, y_train, batch_size=8, validation_split = 0.2, epochs=50)
y_test_pred = model.predict(x_test)

showAccuracyMetrics("Neural Network: ", model, y_test, y_test_pred)
# \"Loss\"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(f'{homedir}/geoweaver_demo/NN_modelLoss.png')

def visualizeResults(modelname, x_test, y_test, pred):
	# Visualization
    ## Check the fitting on training set
	plt.scatter(x_test[:,3], y_test, color='blue')
	plt.scatter(x_test[:,3], pred, color='black')
	#plt.scatter(y_test, pred, color='black')
	plt.title(modelname + ' Fit on testing set')
	plt.xlabel('TROMPOMI-Test')
	plt.ylabel('EPA-Test')
	plt.show()
    
#visualizeResults("Neural Network", x_test, y_test, y_test_pred)
