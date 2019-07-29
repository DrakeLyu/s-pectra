# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 21:39:04 2018

@author: NathanDrake
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 21:39:04 2018

@author: NathanDrake
"""
import re
import keras
import random
import numpy as np
from keras import models
from keras import layers
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.decomposition import PCA
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import average_precision_score
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import scale
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import balanced_accuracy_score
from sklearn.decomposition import NMF
from sklearn.decomposition import FastICA
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import model_selection
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
def load_data():
    y = []
    x =[]
    num_label = 20
    file = open("D:/spectra/nn/x_mix.csv", 'r',newline='')    
    file2 = open("D:/spectra/nn/y_mix.csv", 'r',newline='')

    for line in file:
        data = re.split(',',line)
        for i in range(len(data)):
            data[i] = float(data[i])+5*abs(random.gauss(mu=0,sigma=0.2))
        x.append( data )

    for line in file2:
        y_one = [0]*num_label
        label = line.split(',')
        for lab in label:
            num = int(lab.strip('\r\n').strip('HMDB'))
            y_one[num-1] = 1
        y.append(y_one)
    #print(len(x),len(x[0]))
    return x,y


def plot(history):
        # 绘制训练 & 验证的准确率值
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def train_classifier( x,y ):
    #transformer = PCA()
    #transformer = NMF(n_components=20, init='random', random_state=0)
    transformer = FastICA(n_components=20, random_state=0)
    
    #min_max_scaler = preprocessing.MinMaxScaler()
    #x_minmax = min_max_scaler.fit_transform(x)
    
    A = transformer.fit_transform(x)   
    X = transformer.inverse_transform(A)
    
    x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4 )

    #x_train, x_test, y_train, y_test = train_test_split( X, y , test_size=0.2, random_state=4 )
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.1, patience=80, verbose=1, mode='max')
    callbacks_list = [earlystop]     
    network = models.Sequential()

    neurons = x_train.shape[1]

    network = models.Sequential()
    
##    network.add(layers.Dense(neurons, activation='relu', input_shape=(neurons,)))
##    network.add(BatchNormalization())
##    network.add(layers.Dense(neurons, activation='relu'))
##    network.add(BatchNormalization())
##    network.add(layers.Dense(neurons//2, activation='relu'))
##    network.add(BatchNormalization())
##    network.add(layers.Dense(neurons//2, activation='relu'))
##    network.add(BatchNormalization())
##    network.add(layers.Dense(neurons//5, activation='relu'))
##    network.add(BatchNormalization())
##    network.add(layers.Dense(neurons//10, activation='relu'))
##    network.add(BatchNormalization())    


    x_train = x_train.reshape(-1, 524, 3)
    x_test = x_test.reshape(-1, 524, 3)
    network.add(Conv1D(32,5,activation='relu', padding='same',input_shape=(neurons//3,3)))
    network.add(Conv1D(32,5,activation='relu'))
    #network.add(layers.MaxPooling1D(2))
    network.add(BatchNormalization())
    #network.add(Dropout(0.25))
    network.add(Conv1D(64,3, padding='same',activation='relu'))
    network.add(Conv1D(64,3,activation='relu'))
    #network.add(layers.MaxPooling1D(2))
    #network.add(Dropout(0.25))
    network.add(BatchNormalization()) 
    network.add(layers.Flatten())
    
    network.add(layers.Dense(neurons//5,activation='relu'))
    #network.add(Dropout(0.5))
    network.add(BatchNormalization())
    
    network.add(layers.Dense(neurons//10, activation='relu'))
    network.add(BatchNormalization())
    network.add(layers.Dense(20, activation='sigmoid'))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    #categorical_crossentropy
    network.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

    network.summary()

    network.save_weights("nn_clean_data.h5")

    history = network.fit(x_train, y_train, validation_split=0.2, epochs=150, batch_size=64, callbacks=callbacks_list)
    
    test_loss, test_acc = network.evaluate(x_test, y_test)
        
    pred = network.predict(x_test)

    acc = []
    for i in range(len(y_test)):
        pos = 0
        for j in range(len(y_test[i])):
            if pred[i][j] >=0.5 and y_test[i][j]==1:           
                #pred[i][j]=1
                pos += 1
            elif pred[i][j] <0.5 and y_test[i][j]==0:
                #pred[i][j]=0
                pos += 1
        
        pos /= len(y_test[0])
    
        acc.append(pos)
    print("accuracy:{0:0.2f}".format(np.mean(acc)))

    #plot(history)
    
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(y_test[0])):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            pred[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], pred[:, i])

    
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),pred.ravel())
    average_precision["micro"] = average_precision_score(y_test, pred,average="micro")
    print('Average precision score:{0:0.2f}'.format(average_precision["micro"]))
    print("precision:{0:0.2f}".format(precision["micro"].mean()), "recall:{0:0.2f}".format(recall["micro"].mean()))

    return network


    
x,y =load_data()
network = train_classifier(x,y)

## x
##accuracy: 0.94
##Average precision score: 0.81
##precision: 0.35 recall: 0.85
##
## X ica
##accuracy: 0.95
##Average precision score:0.84
##precision:0.36 recall:0.87

## X nmf
##accuracy:0.95
##Average precision score:0.84
##precision:0.35 recall:0.86

## X pca
##accuracy:0.94
##Average precision score:0.80
##precision:0.34 recall:0.84

## cnn x
##accuracy:0.97
##Average precision score:0.95
##precision:0.49 recall:0.91


## cnn A pca
##accuracy:0.92
##Average precision score:0.64
##precision:0.31 recall:0.78
