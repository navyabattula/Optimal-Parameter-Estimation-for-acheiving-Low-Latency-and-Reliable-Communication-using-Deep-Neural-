# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:26:37 2020

@author: bsr
"""
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
import pandas as pd
data=pd.read_csv('GE-dataset.csv')
training_data=data.drop(["p","r","h","k"],axis=1)
target_data=data[['p','r','h','k']]
model=Sequential()
model.add(Dense(80,input_dim=10000,activation='relu'))
model.add(Dense(80,input_dim=500,activation='relu'))
model.add(Dense(60,input_dim=250,activation='relu'))
model.add(Dense(60,input_dim=250,activation='relu'))
model.add(Dense(40,input_dim=125,activation='relu'))
model.add(Dense(40,input_dim=125,activation='relu'))
model.add(Dense(40,input_dim=80,activation='relu'))
model.add(Dense(40,input_dim=80,activation='relu'))
model.add(Dense(40,input_dim=60,activation='relu'))
model.add(Dense(40,input_dim=60,activation='relu'))
model.add(Dense(20,input_dim=40,activation='relu'))
model.add(Dense(20,input_dim=40,activation='relu'))
model.add(Dense(20,input_dim=20,activation='relu'))
model.add(Dense(20,input_dim=20,activation='relu'))
model.add(Dense(10,input_dim=10,activation='relu'))
model.add(Dense(10,input_dim=10,activation='relu'))
model.add(Dense(4,activation='sigmoid'))
model.compile(loss='mean_squared_error',
             optimizer='adam',
             metrics=['accuracy'])
model.fit(training_data,target_data,epochs=100,verbose=2)
b= np.zeros([12,000,4], dtype=float)
b=((model.predict(training_data).round(1)-target_data))
x= np.sum(b)
z=x/target_data
p=np.min(z)
q=np.max(z)
print('Normalized Mean square error is')
print(p,q)