####################################################################
# main.py
####################################################################
# main is used to execute the code and run experiments.
####################################################################

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import time

#import cupy as cp

from classifier import Classifier
from dimreduce import DimReduce
import results

def main():
    np.random.seed() # Consistency

    try:
        data = np.load("Data/DASHlink_full_fourclass_raw_comp.npz")
    except:
        print("Data/DASHlink_full_fourclass_raw_comp.npz not found! Please download the file from https://c3.ndc.nasa.gov/dashlink/resources/1018/")
        return
    
    X = data['data'] # 99837 samples, 20 features over 160 seconds (1 second sampling rate), technically 3200 features total
    y = data['label'] # 99837 labels, [89663, 7013, 2207, 954] distribution

    flatX = X.reshape(X.shape[0], X.shape[1] * X.shape[2]) # Turns the 2D data into 1D data

    X_train, X_rest, y_train, y_rest = train_test_split(flatX, y, test_size=.4) # 60% training data
    X_valid, X_test, y_valid, y_test = train_test_split(X_rest, y_rest, test_size=.5) # 20% validation data, 20% testing data

    # Dimensionality Reduction
    t0 = time.time()

    # none , pca , lda , tsvd, fica, isomap, autoencoder
    dim_reduce = DimReduce("fica", 3)
    dim_reduce.fit(X_train, y_train)

    t1 = time.time()


    dimreductime = t1-t0
    dX_train = dim_reduce.reduce(X_train)
    dX_valid = dim_reduce.reduce(X_valid)
    dX_test = dim_reduce.reduce(X_test)

    # X_train = cp.array(X_train)
    # y_train = cp.array(y_train)
    # X_valid = cp.array(X_valid)
    # X_test = cp.array(X_test)
    # y_valid = cp.array(y_valid)
    # y_test = cp.array(y_test)

    # Classification
    
    model = Classifier(estimators=4, depth=20, lr=1.0)

    t2 = time.time()
    best_params = model.optimize(dX_train, y_train, dX_valid, dX_test, y_valid, y_test)
    t3 = time.time()

    t4 = time.time()
    model.train(dX_train, y_train, dX_valid, dX_test, y_valid, y_test, best_params)
    t5 = time.time()
    

    preds, acc = model.test(dX_valid, y_valid)

    # model.train(X_train, y_train, X_valid, X_test, y_valid, y_test)
    # preds, acc = model.test(X_valid, y_valid)

    
    optimizetime = t3-t2

    xgbtime = t5-t4

    
    results.accuracy(y_valid,preds,acc)
    print("Time to run dimensionality reduction " + str(dimreductime))
    print("Time spent optimizing parameters " + str(optimizetime))
    print("Time to train and test XGB " + str(xgbtime))

if __name__ == "__main__":
    main()
