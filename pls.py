# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 21:39:04 2018

@author: NathanDrake
"""
import re,math
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras import models
from keras import layers
from keras.utils import plot_model
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error
def load_data():
    items = ["nap","pyr","phe"]
    y = {item:[] for item in items}
    x =[]
    path = "D:/spectra/raman/clean_data.csv"
    
    file = open(path, 'r',newline='')
    line = file.readline()
    for line in file:
        #label = []
        data = re.split(',',line)
        for i in range(len(data)):
            data[i] = float(data[i])
            if i == 2:
                y["nap"].append(data[2])
            if i == 4:
                y["pyr"].append(data[4])                
            if i == 6:
                y["phe"].append(data[6])

        x.append( data[7:] )


        
    return x,y



def Partial_Least_Squares(x,y):
    mse = []
    score = []
    for i in ["nap","pyr","phe"]:
        
        x_train, x_test, y_train, y_test = train_test_split( x, y[i], test_size=0.2, random_state=4 )
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        n = len(x_train)
        mse = []

        for j in np.arange(1, 20):

            pls = PLSRegression(n_components=j)
            pls.fit(scale(x_train), y_train)

            score = -model_selection.cross_val_score(pls, scale(x_train), y_train, cv=kf_10, scoring='neg_mean_squared_error').mean()
            #score = model_selection.cross_val_score(pls, scale(x_train), y_train, cv=kf_10, scoring='r2').mean()

            mse.append(math.sqrt(score))

        # Plot results
        plt.plot(np.arange(1, 20), np.array(mse), '-v')
        plt.xlabel('Number of Partial_Least_Squares components in regression')
        plt.ylabel('RMSE')
        plt.title(i)
        plt.xlim(xmin=-1)
        plt.show()

    
def Principal_Components_Regression(x,y):
    pca2 = PCA()
    # 10-fold CV, with shuffle

    for i in ["nap","pyr","phe"]:
        
        x_train, x_test, y_train, y_test = train_test_split( x, y[i], test_size=0.2, random_state=4 )
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)


        # Scale the data
        X_reduced_train = pca2.fit_transform(scale(x_train))
        n = len(X_reduced_train)


        mse = []
        regr = LinearRegression()
        
        # Calculate MSE with only the intercept (no principal components in regression)
        score = -1*model_selection.cross_val_score(regr, np.ones((n,1)), y_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()    
        mse.append(math.sqrt(score))

        # Calculate MSE using CV for the 19 principle components, adding one component at the time.
        for j in np.arange(1, 20):
            score = -1*model_selection.cross_val_score(regr, X_reduced_train[:,:j], y_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()
            #mse.append(score)
            mse.append(math.sqrt(score))
            
        plt.plot(np.array(mse), '-v')
        plt.xlabel('Number of principal components in regression')
        plt.ylabel('RMSE')
        plt.title(i)
        plt.xlim(xmin=-1)
        plt.show()


    
def Support_Vector_Regression(x,y):

    mse = []
    score = []
    for i in ["nap","pyr","phe"]:
        
        x_train, x_test, y_train, y_test = train_test_split( x, y[i], test_size=0.2, random_state=4 )
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        n = len(x_train)
        mse = []

        svr = SVR(kernel='rbf',gamma='scale', C=1, epsilon=0.2)
        svr.fit(scale(x_train), y_train)

        score = -model_selection.cross_val_score(svr, scale(x_train), y_train, cv=kf_10, scoring='neg_mean_squared_error').mean()
        #score = model_selection.cross_val_score(svr, scale(x_train), y_train, cv=kf_10, scoring='r2').mean()

        mse.append(math.sqrt(score))

        # The "accuracy" scoring is proportional to the number of correct
        # classifications
        rfecv = RFECV(estimator=svr, step=1, cv=StratifiedKFold(2),scoring='accuracy')
        rfecv.fit(x_train, y_train)

        print("Optimal number of features : %d" % rfecv.n_features_)

        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()
            
##        # Plot results
##        plt.plot( np.array(mse), '-v')
##        plt.xlabel('Number of SVR components in regression')
##        plt.ylabel('RMSE')
##        plt.title(i)
##        plt.xlim(xmin=-1)
##        plt.show()

def train_classifier( x,y ):

    mse = []

    for i in ["nap","pyr","phe"]:
        
        x_train, x_test, y_train, y_test = train_test_split( x, y[i], test_size=0.2, random_state=4 )
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        n = len(x_train)


        network = models.Sequential()

        neurons = x_train.shape[1]

        network.add(layers.Dense(neurons, activation='relu', input_shape=(neurons,)))

        network.add(layers.Dense(neurons//2, activation='relu'))

        network.add(layers.Dense(neurons//5, activation='relu'))

        network.add(layers.Dense(neurons//5, activation='relu'))
        
        network.add(layers.Dense(neurons//10, activation='relu'))

        network.add(layers.Dense(neurons//10, activation='relu'))

        network.add(layers.Dense(neurons//20, activation='relu'))

        network.add(layers.Dense(neurons//50, activation='relu'))
        
        network.add(layers.Dropout(0.5))

        network.add(layers.Dense(1, activation='sigmoid'))

        #categorical_crossentropy
        network.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
        network.compile(optimizer='adam',loss='mean_absolute_error',metrics=['accuracy'])
        network.summary()

        network.save_weights("nn_clean_data.h5")

        history = network.fit(scale(x_train), y_train, validation_split=0.15, epochs=150, batch_size=64)
            
        test_loss, test_acc = network.evaluate(x_test, y_test)

        pred = network.predict(x_test)
        print(pred)
        mse0 = mean_squared_error(y_test, pred)
        mse.append(mse0)

    print(mse)




    #plot_model(network, to_file='model.png')

    return network



x,y = load_data()

# 10-fold CV, with shuffle
kf_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

#Partial_Least_Squares(x,y)
#Principal_Components_Regression(x,y)
Support_Vector_Regression(x,y)

#train_classifier(x,y)
