import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD,Adam
from sklearn.datasets import load_boston
import csv



x = np.loadtxt('/Users/zhaozining/Desktop/house_price/Dataset.csv',delimiter = ",",skiprows = 1)
y = np.loadtxt('/Users/zhaozining/Desktop/house_price/Target.csv',delimiter = ",", skiprows = 1,usecols =0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

mean = x.mean(axis=0)
std = x.std(axis=0)
x_train -= mean
x_train /=std
x_test -= mean
x_test /= std

def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu',input_shape=(x_train.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    opt = Adam(lr = 0.01)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return model

num_epochs = 100

model = build_model()
history = model.fit(x_train, y_train, epochs=num_epochs*20, batch_size=50, verbose=1)
pre_data = model.predict(x_test)

model.summary()

plt.figure(1)
plt.plot(y_test, label='real data')
plt.plot(pre_data, label='forecasting data')
plt.xlabel('epochs')
plt.ylabel('housing price')
plt.title('predict housing price')
plt.legend()

plt.figure(2)
plt.plot(history.history['loss'])
plt.title('loss')

plt.figure(3)
plt.plot(history.history['mae'])
plt.title('mean absolute error')
plt.show()


model.save('/Users/zhaozining/Desktop/house_price/boston_housing.h5')
print('model saved')
