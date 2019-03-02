# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 21:39:04 2018

@author: NathanDrake
"""

import keras
import numpy as np
from keras import models
from keras import layers

def load_data():

    train_set_x_orig=np.loadtxt(open("x_train.csv","rb"),dtype=float,delimiter=',',skiprows=0)
     
    train_set_y_orig=np.loadtxt(open("y_train.csv","rb"),dtype=int,delimiter=',',skiprows=0)# your train set labels    

    test_set_x_orig=np.loadtxt(open("x_test.csv","rb"),dtype=float,delimiter=',',skiprows=0) # your test set features
    
    test_set_y_orig=np.loadtxt(open("y_test.csv","rb"),dtype=int,delimiter=',',skiprows=0) # your test set labels

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig

train_x, train_y, test_x, test_y = load_data()
#train_x= train_x.T
#train_y= train_y.T
print(train_x.shape)
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(124,)))

network.add(layers.Dense(256, activation='relu'))

network.add(layers.Dense(128, activation='relu'))

network.add(layers.Dense(50, activation='relu'))

network.add(layers.Dense(50, activation='relu'))

network.add(layers.Dropout(0.5))

network.add(layers.Dense(20, activation='sigmoid'))
#categorical_crossentropy
network.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
network.summary()
network.save("nn.h5")
network.fit(train_x, train_y, epochs=350, batch_size=128)
test_loss, test_acc = network.evaluate(test_x, test_y)
predictions = network.predict(test_x)
print(test_acc)


