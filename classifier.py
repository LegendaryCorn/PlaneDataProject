####################################################################
# classifier.py
####################################################################
# Contains xgboost. Can process training and testing sets.
####################################################################

from xgboost import XGBClassifier
import numpy as np

class Classifier:
    def __init__(self, estimators, depth, lr):
        self.bst = XGBClassifier(n_estimators=estimators, max_depth=depth, learning_rate=lr, objective='binary:logistic')
    
    def train(self, X_train, y_train):
        self.bst.fit(X_train, y_train)

    def test(self, X_test, y_test):
        preds = self.bst.predict(X_test)
        acc = np.equal(y_test, preds)
        return preds, acc
        print(y_test)
        print(preds)