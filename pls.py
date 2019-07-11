import re,math
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras import models
from keras import layers
from keras.layers import Conv1D
from keras.layers import Dropout
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
from sklearn.metrics import mean_absolute_error
from keras.optimizers import Adam
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from keras.callbacks import EarlyStopping
def load_data():
    items = ["nap","pyr","phe"]
    y = {item:[] for item in items}
    x =[]
    path = "C:/Users/jiali/Desktop/clean_data.csv"
    
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
    rmsec = []
    rmsep = []
    rmsecv = []
    Y = []
    for j in range(300):
        m = []
        for i in ["nap","pyr","phe"]:
            m.append( y[i][j])
        Y.append(m)

    #for i in ["nap","pyr","phe"]:
    if True:        
        x_train, x_test, y_train, y_test = train_test_split( scale(x), Y, test_size=0.1, random_state=4 )
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        n = len(x_train)
        mse = []

##        for j in np.arange(1, 30):
##
##            pls = PLSRegression(n_components=j)
##            pls.fit(scale(x_train), y_train)
##
##            score = -model_selection.cross_val_score(pls, scale(x_train), y_train, cv=kf_10, scoring='neg_mean_squared_error').mean()
##            #score = model_selection.cross_val_score(pls, scale(x_train), y_train, cv=kf_10, scoring='r2').mean()
##            mse.append(math.sqrt(score))
            
        # j=5: rmse: 2.45 1.15 2.15
        pls = MultiOutputRegressor(PLSRegression(n_components=5))
        pls.fit(x_train, y_train)

####        pred = pls.predict(scale(x_test) )       
####        mse0 = mean_squared_error(y_test, pred)
####        rmsep.append(math.sqrt(mse0))
####        
####        pred1 = pls.predict(scale(x_train)  )
####        mse1 = mean_squared_error(y_train, pred1)
####        rmsec.append(math.sqrt(mse1))
####
####        score = -1*model_selection.cross_val_score(pls, x_train, y_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()    
####        rmsecv.append(math.sqrt(score))

        pred0 = pls.predict(x_train )        
        pred = pls.predict(x_test)

        for i in range(len(y_test)):
            for j in range(0,3):
                if pred[0][i][j] <2:
                    pred[0][i][j] = 0
        for i in range(len(y_test)):
            print(y_test[i],pred[0][i])
        for i in range(0,3):

            mse0 = mean_squared_error(y_test[:,i], pred[0,:,i])
            rmsep.append(math.sqrt(mse0))

            
        for i in range(0,3):
               
            mse1 = mean_squared_error(y_train[:,i], pred0[0,:,i])
            rmsec.append(math.sqrt(mse1))
            
    print("rmsec",rmsec)
    #print("rmsecv",rmsecv)
    print("rmsep",rmsep)
##rmsec [1.797274515117025, 0.9091525234438878, 1.6283716396300185]
##rmsecv [2.44657116400538, 1.1688601755591697, 2.1456749315348964]
##rmsep [2.0503465798988847, 1.0905545210511391, 1.9545315544624196]

        # Plot results
##        plt.plot(np.arange(1, 30), np.array(mse), '-v')
##        plt.xlabel('Number of Partial_Least_Squares components in regression')
##        plt.ylabel('RMSE')
##        plt.title(i)
##        plt.xlim(xmin=-1)
##        plt.show()

    
def Principal_Components_Regression(x,y):
    pca2 = PCA()
    # 10-fold CV, with shuffle

    for i in ["nap","pyr","phe"]:
        
        x_train, x_test, y_train, y_test = train_test_split( x, y[i], test_size=0.1, random_state=4 )
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
        #score = -1*model_selection.cross_val_score(regr, np.ones((n,1)), y_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()    
        #mse.append(math.sqrt(score))

        # Calculate MSE using CV for the 19 principle components, adding one component at the time.
        for j in np.arange(1, 500):
            score = -1*model_selection.cross_val_score(regr, X_reduced_train[:,:j], y_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()
            #mse.append(score)
            mse.append(math.sqrt(score))
            
        plt.plot(np.array(mse), '-v')
        plt.xlabel('Number of principal components in regression')
        plt.ylabel('RMSE')
        plt.title(i)
        plt.xlim(xmin=-1)
        plt.show()

    
def PCA_SVR(x,y):
    pca2 = PCA()
    kpca = KernelPCA(n_components=500, kernel='poly')
    # 10-fold CV, with shuffle
    rmsec = []
    rmsep = []
    rmsecv = []
    maep = []
    maec = []
    Y = []
    for j in range(300):
        m = []
        for i in ["nap","pyr","phe"]:
            m.append( y[i][j])
        Y.append(m)
    if True:
        # Scale the data
        x_reduced = pca2.fit_transform(scale(x))        
        x_train, x_test, y_train, y_test = train_test_split( x_reduced, Y, test_size=0.1, random_state=4 )
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        n =300

        #svr = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,max_depth=2, random_state=0, loss='ls'))
        svr = MultiOutputRegressor(SVR(kernel='sigmoid',gamma='scale', C=1, epsilon=0.2))

        svr.fit(x_train[:,:n], y_train)
        
        pred0 = svr.predict(x_train[:,:n] )        
        pred = svr.predict(x_test[:,:n])

##        for i in range(len(pred)):
##            for j in range(0,3):
##                if pred[i][j] < 2:
##                    pred[i][j] = 0

        for i in range(len(y_test)):
            print(y_test[i],pred[i])
            
        for i in range(0,3):
            mse0 = mean_squared_error(y_test[:,i], pred[:,i])
            mae0 = mean_absolute_error(y_test[:,i], pred[:,i])
            rmsep.append(math.sqrt(mse0))
            maep.append(mae0)
            
        for i in range(0,3):
            mse1 = mean_squared_error(y_train[:,i], pred0[:,i])
            mae1 = mean_absolute_error(y_train[:,i], pred0[:,i])
            rmsec.append(math.sqrt(mse1))
            maec.append(mae1)
            
    print("rmsec",rmsec)
    #print("rmsecv",rmsecv)
    print("rmsep",rmsep)
    print("maec",maec)
    print("maep",maep)
    print(len(pred))
    m = np.arange(0,30)

    fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    plt.ylabel('Concentration/ppb')    
    ax[0].scatter(m,pred[:,0],c='tab:blue',label='y_predict')
    ax[0].scatter(m,y_test[:,0],c='tab:orange',label='y_test')
    ax[0].vlines(m,[0],y_test[:,0])
    ax[0].set_title('BaP')
    ax[0].legend()
    ax[1].scatter(m,pred[:,1],c='tab:blue',label='y_predict')
    ax[1].scatter(m,y_test[:,1],c='tab:orange',label='y_test')
    ax[1].vlines(m,[0],y_test[:,1])
    ax[1].set_title('PHE')
    ax[1].legend()
    ax[2].scatter(m,pred[:,2],c='tab:blue',label='y_predict')
    ax[2].scatter(m,y_test[:,2],c='tab:orange',label='y_test')
    ax[2].vlines(m,[0],y_test[:,2])
    ax[2].set_title('PYR')
    ax[2].legend()
    fig.suptitle('Predicted value and true value')

    #ax.grid(True)
    plt.show()

        # Calculate MSE using CV for the 19 principle components, adding one component at the time.

##        mse = []
##        for j in np.arange(1, 500):
##            score = -1*model_selection.cross_val_score(svr, x_train[:,:j], y_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()
##            mse.append(math.sqrt(score))
##            
##        plt.plot(np.array(mse), '-v')
##        plt.xlabel('Number of principal components in SVRegression')
##        plt.ylabel('RMSE')
##        plt.title(i)
##        plt.xlim(xmin=-1)
##        plt.show()
        


##rmsec [2.243769480029148, 1.1062405845703978, 1.8390116321584529]
##rmsecv [2.8801699992822853, 1.5907525689713935, 2.4536471382399325]
##rmsep [2.552785927048447, 1.0506031209984712, 2.2216336226674955]
    
def Support_Vector_Regression(x,y):

    rmsec = []
    rmsep = []
    rmsecv = []

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

        pred = svr.predict(scale(x_test)  )      
        mse0 = mean_squared_error(y_test, pred)
        rmsep.append(math.sqrt(mse0))
        
        pred1 = svr.predict(scale(x_train) )     
        mse1 = mean_squared_error(y_train, pred1)
        rmsec.append(math.sqrt(mse1))

        score = -1*model_selection.cross_val_score(svr, scale(x_train), y_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()    
        rmsecv.append(math.sqrt(score))

    print("rmsec",rmsec)
    print("rmsecv",rmsecv)
    print("rmsep",rmsep)
##rmsec [2.368523885910574, 0.991257964247443, 1.8731340159628915]
##rmsecv [2.8774545358841026, 1.3524005798781358, 2.3626260475960232]
##rmsep [2.473416164122062, 1.358156026927806, 2.2385410445977505]
    
##        # The "accuracy" scoring is proportional to the number of correct
##        # classifications
##        rfecv = RFECV(estimator=svr, step=1, cv=StratifiedKFold(2),scoring='accuracy')
##        rfecv.fit(x_train, y_train)
##
##        print("Optimal number of features : %d" % rfecv.n_features_)
##
##        # Plot number of features VS. cross-validation scores
##        plt.figure()
##        plt.xlabel("Number of features selected")
##        plt.ylabel("Cross validation score (nb of correct classifications)")
##        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
##        plt.show()

       
##        # Plot results
##        plt.plot( np.array(mse), '-v')
##        plt.xlabel('Number of SVR components in regression')
##        plt.ylabel('RMSE')
##        plt.title(i)
##        plt.xlim(xmin=-1)
##        plt.show()



def ANN(x,y):
    pca2=PCA()
    rmsec = []
    rmsep = []
    rmsecv = []
    maep = []
    maec = []
    Y = []
    for j in range(300):
        m = []
        for i in ["nap","pyr","phe"]:
            m.append( y[i][j])
        Y.append(m)

    if True:
        x_reduced = pca2.fit_transform(scale(x))
        #x_train, x_test, y_train, y_test = train_test_split( scale(x), Y, test_size=0.1, random_state=4 )

        x_train, x_test, y_train, y_test = train_test_split( x_reduced[:,:250],Y , test_size=0.15, random_state=4 )
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        n = len(x_train)
        earlystop = EarlyStopping(monitor='val_acc', min_delta=0.1, patience=50,verbose=1, mode='auto')
        callbacks_list = [earlystop]       
        network = models.Sequential()

        neurons = x_train.shape[1]

        network.add(layers.Dense(neurons*2, activation='relu', input_shape=(neurons,)))
        network.add(BatchNormalization())
        
        network.add(layers.Dense(neurons*2, activation='relu'))
        network.add(BatchNormalization())
        
        network.add(layers.Dense(neurons, activation='relu'))
        network.add(BatchNormalization())
        
        network.add(layers.Dense(neurons, activation='relu'))
        network.add(BatchNormalization())
        
        network.add(layers.Dense(neurons//2, activation='relu'))
        network.add(BatchNormalization())        

        network.add(layers.Dense(3, activation='relu'))

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        network.compile(optimizer=adam,loss='mean_squared_error',metrics=['accuracy'])
        #network.compile(optimizer = adam,loss='mean_absolute_error',metrics=metrics=[metrics.mae])
        network.summary()
        history = network.fit(x_train, y_train, validation_split=0.1, epochs=300, batch_size=64,callbacks=callbacks_list)

        pred0 = network.predict(x_train )        
        pred = network.predict(x_test)
        for i in range(len(y_test)):
            print(y_test[i],pred[i])
            
        for i in range(0,3):
            mse0 = mean_squared_error(y_test[:,i], pred[:,i])
            mae0 = mean_absolute_error(y_test[:,i], pred[:,i])
            rmsep.append(math.sqrt(mse0))
            maep.append(mae0)
            
        for i in range(0,3):
            mse1 = mean_squared_error(y_train[:,i], pred0[:,i])
            mae1 = mean_absolute_error(y_train[:,i], pred0[:,i])
            rmsec.append(math.sqrt(mse1))
            maec.append(mae1)
            
        print("rmsec",rmsec)
        #print("rmsecv",rmsecv)
        print("rmsep",rmsep)
        print("maec",maec)
        print("maep",maep)
        print(len(pred))

    def error_plot():
        m = np.arange(0,45)

        fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
        plt.ylabel('Concentration/ppb')    
        ax[0].scatter(m,pred[:,0],c='tab:blue',label='y_predict')
        #ax[0].scatter(m,y_test[:,0],c='tab:orange',label='y_test')
        ax[0].vlines(m,[0],y_test[:,0],label='y_test')
        ax[0].set_title('BaP')
        ax[0].legend()
        ax[1].scatter(m,pred[:,1],c='tab:blue',label='y_predict')
        #ax[1].scatter(m,y_test[:,1],c='tab:orange',label='y_test')
        ax[1].vlines(m,[0],y_test[:,1],label='y_test')
        ax[1].set_title('PHE')
        ax[1].legend()
        ax[2].scatter(m,pred[:,2],c='tab:blue',label='y_predict')
        #ax[2].scatter(m,y_test[:,2],c='tab:orange',label='y_test')
        ax[2].vlines(m,[0],y_test[:,2],label='y_test')
        ax[2].set_title('PYR')
        ax[2].legend(loc='upper right')
        fig.suptitle('Predicted value and true value')

        #ax.grid(True)
        plt.show()

    def training_plot():
            # Plot training & validation accuracy values
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.show()

            # Plot training & validation loss values
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.show()
    error_plot()            
    training_plot()
##rmsec [0.8868594169441504, 0.4694611642639408, 0.7505665612748366]
##rmsep [1.8627601181173121, 0.9917311552473371, 1.7557357197799104]
##pca:
##rmsec [0.9253798976363444, 0.48194183390650663, 0.8184379164513504]
##rmsep [1.9398267375059182, 1.0201064593767042, 1.7130587826371526]
    
def CNN(x,y):
    pca2=PCA()
    rmsec = []
    rmsep = []
    rmsecv = []
    Y = []
    
    for j in range(300):
        m = []
        for i in ["nap","pyr","phe"]:
            m.append( y[i][j])
        Y.append(m)
    
    x_reduced = pca2.fit_transform(scale(x))
    #x_train, x_test, y_train, y_test = train_test_split( x, y[i], test_size=0.1, random_state=4 )

    x_train, x_test, y_train, y_test = train_test_split( x, Y, test_size=0.15, random_state=4 )
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    n = len(x_train)
    neurons = x_train.shape[1]
    
    network = models.Sequential()
    network.add(Conv1D(32,neurons,activation='relu', padding='same',input_shape=(neurons,1)))
    network.add(Conv1D(32,neurons,activation='relu'))
    #network.add(layers.MaxPooling1D(neurons))
    network.add(Dropout(0.25))

    network.add(Conv1D(64,neurons, padding='same',activation='relu'))
    network.add(Conv1D(64,neurons,activation='relu'))
    #network.add(layers.MaxPooling1D(neurons))
    network.add(Dropout(0.25))

    network.add(layers.Flatten())
    network.add(layers.Dense(neurons),activation='relu')
    network.add(Dropout(0.5))
    network.add(layers.Dense(3),activation='relu')

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    network.compile(optimizer=adam,loss='mean_squared_error',metrics=['accuracy'])

    network.summary()
    history = network.fit(x_train, y_train, validation_split=0.1, epochs=200, batch_size=64)
    
    pred = network.predict(x_test )    
    mse0 = mean_squared_error(y_test, pred)
    rmsep.append(math.sqrt(mse0))
        
    pred1 = network.predict(x_train )     
    mse1 = mean_squared_error(y_train, pred1)
    rmsec.append(math.sqrt(mse1))

    print("rmsec",rmsec)
    #print("rmsecv",rmsecv)
    print("rmsep",rmsep)
   
x,y = load_data()

# 10-fold CV, with shuffle
kf_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

#Partial_Least_Squares(x,y)
#Principal_Components_Regression(x,y)
#Support_Vector_Regression(x,y)
#PCA_SVR(x,y)
ANN(x,y)
#CNN(x,y)
