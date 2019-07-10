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

    sel = VarianceThreshold(threshold=100000)
    x_vt = sel.fit_transform(x)
    print(len(x[0]))
    print(len(x_vt[0]))
    print(x_vt)
    return x,y

def main():
    Y = []
    x,y = load_data()
    for j in range(300):
        m = []
        for i in ["nap","pyr","phe"]:
            m.append( y[i][j])
        Y.append(m)
    if True:    
        x_train, x_test, y_train, y_test = train_test_split( scale(x), Y, test_size=0.1, random_state=4 )
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

    ##etimator = LinearRegression()
    estimator = MultiOutputRegressor(SVR(kernel='rbf',gamma='scale', C=1, epsilon=0.2))

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
    selector = estimator.fit(x_train,y_train[:,0])

    print(selector.support_)

    x_reduced = []
    for j in range(len(X)):
        row = []
        for i in range(len(selector.support_)):
            if selector.support_[i] ==True:
                row.append(X[j][i])
        x_reduced.append(row)

##
##        n =300
##
##    svr = MultiOutputRegressor(SVR(kernel='rbf',gamma='scale', C=1, epsilon=0.2))
##    svr.fit(x_train, y_train)
##        
    pred0 = selector.predict(x_train )        
    pred = selector.predict(x_test )
    print(pred)
##    for i in range(len(pred)):
##        for j in range(0,3):
##            if pred[i][j] < 2:
##                pred[i][j] = 0
##
##    for i in range(len(y_test)):
##        print(y_test[i],pred[i])
##            
##    for i in range(0,3):
##        mse0 = mean_squared_error(y_test[:,i], pred[:,i])
##        rmsep.append(math.sqrt(mse0))
##
##            
##    for i in range(0,3):
##        mse1 = mean_squared_error(y_train[:,i], pred0[:,i])
##        rmsec.append(math.sqrt(mse1))
##            
##    print("rmsec",rmsec)
##    #print("rmsecv",rmsecv)
##    print("rmsep",rmsep)
##    
if __name__ == "__main__":
    main()
