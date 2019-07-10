from __future__ import print_function
import numpy as np
from sklearn import datasets, linear_model
import re,math
from sklearn.linear_model import LinearRegression
from genetic_selection import GeneticSelectionCV
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
from sklearn.feature_selection import VarianceThreshold

def load_data():
    wl = []
    items = ["nap","pyr","phe"]
    y = {item:[] for item in items}
    x =[]
    Y = []
    path = "C:/Users/jiali/Desktop/clean_data.csv"
    
    file = open(path, 'r',newline='')
    line = file.readline()

    for i in re.split(',',line)[7:]:
        wl.append(float(i.strip('\n')))

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
    for j in range(300):
        m = []
        for i in ["nap","pyr","phe"]:
            m.append( y[i][j])
        Y.append(m)
    #sel = VarianceThreshold(threshold=100000)
    #x_vt = sel.fit_transform(x)

    return x,y, Y,wl

def ga():

    x,y,Y = load_data()
    index = []

    for i in ["nap","pyr","phe"]:
        estimator = LinearRegression()
        ##estimator = SVR(kernel='rbf',gamma='scale', C=1, epsilon=0.2)
        selector = GeneticSelectionCV(estimator,
                                      cv=5,
                                      verbose=1,
                                      scoring="accuracy",
                                      max_features=5,
                                      n_population=100,
                                      crossover_proba=0.5,
                                      mutation_proba=0.2,
                                      n_generations=100,
                                      crossover_independent_proba=0.5,
                                      mutation_independent_proba=0.05,
                                      tournament_size=3,
                                      n_gen_no_change=10,
                                      caching=True,
                                      n_jobs=-1)
        selector = selector.fit(x,y[i])

        print(selector.support_)

        for j in range(len(selector.support_)):
            if selector.support_[j] ==True:
                if j not in index:
                    index.append(j)
    index.sort()
    
    x_reduced = []
    for i in range(len(x)):
        row = []
        for j in range(len(x[0])):
            if j in index:
                row.append(x[i][j])
        x_reduced.append(row)
    print(len(x_reduced))
    print(len(x_reduced[0]))

    return x,x_reduced,Y

def svr(x,y):
    rmsec = []
    rmsep = []
    x_train, x_test, y_train, y_test = train_test_split( scale(x), Y, test_size=0.1, random_state=4 )
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    ##        
    svr = MultiOutputRegressor(SVR(kernel='rbf',gamma='scale', C=1, epsilon=0.2))
    svr.fit(x_train, y_train)
    pred0 = svr.predict(x_train)
    pred = svr.predict(x_test) 
    ##        
    ##
    ##    for i in range(len(pred)):
    ##        for j in range(0,3):
    ##            if pred[i][j] < 2:
    ##                pred[i][j] = 0
    ##
    ##    for i in range(len(y_test)):
    ##        print(y_test[i],pred[i])

    for i in range(0,3):
        mse0 = mean_squared_error(y_test[:,i], pred[:,i])
        rmsep.append(math.sqrt(mse0))

                
    for i in range(0,3):
        mse1 = mean_squared_error(y_train[:,i], pred0[:,i])
        rmsec.append(math.sqrt(mse1))
    ##            
    print("rmsec",rmsec)
    ##    #print("rmsecv",rmsecv)
    print("rmsep",rmsep)

##x,x_reduced,Y = ga()
##svr(x,Y)
##svr(x_reduced,Y)
import matplotlib.pyplot as plt

x,y,Y,wl = load_data()

fig,ax1 = plt.subplots()
ax1.plot(wl,x[0])
ax1.set_xlim(350,1800)
ax1.set_xlabel('wavelenth')
ax1.set_ylabel('s1 and s2')
ax1.grid(True)

plt.show()
