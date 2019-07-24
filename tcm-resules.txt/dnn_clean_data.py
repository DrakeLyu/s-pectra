# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 21:39:04 2018

@author: NathanDrake
"""
import re
import keras
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
from sklearn.decomposition import NMF
from sklearn.decomposition import FastICA
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import model_selection
def load_data():
    y = []
    x =[]
    num_label = 20
    file = open("D:/spectra/nn/x_mix.csv", 'r',newline='')    
    file2 = open("D:/spectra/nn/y_mix.csv", 'r',newline='')

    for line in file:
        data = re.split(',',line)
        for i in range(len(data)):
            data[i] = float(data[i])
        x.append( data )

    for line in file2:
        y_one = [0]*num_label
        label = line.split(',')
        for lab in label:
            num = int(lab.strip('\r\n').strip('HMDB'))
            y_one[num-1] = 1
        y.append(y_one)
    print(len(x),len(x[0]))
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
    pca2 = PCA()
    nmf = NMF(n_components=500, init='random', random_state=0)
    transformer = FastICA(n_components=800, random_state=0)    
    min_max_scaler = preprocessing.MinMaxScaler()
    x_minmax = min_max_scaler.fit_transform(x)
    
    x_reduced = transformer.fit_transform(x_minmax)
    #x_reduced = nmf.fit_transform(x_minmax)
    #x_reduced2 = pca2.fit_transform(x_reduced)
    
    #x_train, x_test, y_train, y_test = train_test_split( scale(x), y, test_size=0.2, random_state=4 )

    x_train, x_test, y_train, y_test = train_test_split( x_reduced, y , test_size=0.2, random_state=4 )
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.1, patience=100, verbose=1, mode='max')
    callbacks_list = [earlystop]     
    network = models.Sequential()

    neurons = x_train.shape[1]

    network.add(layers.Dense(neurons, activation='relu', input_shape=(neurons,)))
    network.add(BatchNormalization())
    network.add(layers.Dense(neurons, activation='relu'))
    network.add(BatchNormalization())
    network.add(layers.Dense(neurons//2, activation='relu'))
    network.add(BatchNormalization())
    network.add(layers.Dense(neurons//2, activation='relu'))
    network.add(BatchNormalization())
    network.add(layers.Dense(neurons//5, activation='relu'))
    network.add(BatchNormalization())
    network.add(layers.Dense(neurons//10, activation='relu'))
    network.add(BatchNormalization())    
    #network.add(layers.Dropout(0.5))

    network.add(layers.Dense(20, activation='sigmoid'))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    #categorical_crossentropy
    network.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

    network.summary()

    network.save_weights("nn_clean_data.h5")

    history = network.fit(x_train, y_train, validation_split=0.15, epochs=150, batch_size=64,callbacks=callbacks_list)
    
    test_loss, test_acc = network.evaluate(x_test, y_test)
        
    pred = network.predict(x_test)
    
##    for i in range(len(y_test)):
##        for j in range(len(y_test[i])):
##            if pred[i][j] >0.5:           
##                pred[i][j]=1
##            else:
##                pred[i][j]=0                

##        print(y_test[i],pred[i])

    plot(history)
    
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(y_test[0])):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            pred[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], pred[:, i])
        print(average_precision[i])
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
        pred.ravel())
    average_precision["micro"] = average_precision_score(y_test, pred,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))


    return network


    
x,y =load_data()
network = train_classifier(x,y)

