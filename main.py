####################################################################
# main.py
####################################################################
# main is used to execute the code and run experiments.
####################################################################

import numpy as np
from sklearn.model_selection import train_test_split

from classifier import Classifier
import results

def main():
    np.random.seed(0) # Consistency

    try:
        data = np.load("Data/DASHlink_full_fourclass_raw_comp.npz")
    except:
        print("Data/DASHlink_full_fourclass_raw_comp.npz not found! Please download the file from https://c3.ndc.nasa.gov/dashlink/resources/1018/")
        return
    
    X = data['data'] # 99837 samples, 20 features over 160 seconds (1 second sampling rate), technically 3200 features total
    y = data['label'] # 99837 labels, [89663, 7013, 2207, 954] distribution

    flatX = X.reshape(X.shape[0], X.shape[1] * X.shape[2]) # Turns the 2D data into 1D data

    X_train, X_test, y_train, y_test = train_test_split(flatX, y, test_size=.2)

    model = Classifier(estimators=2, depth=4, lr=1)
    model.train(X_train, y_train)
    preds, acc = model.test(X_test, y_test)
    results.accuracy(y_test,preds,acc)

if __name__ == "__main__":
    main()