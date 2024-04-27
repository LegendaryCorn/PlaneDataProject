####################################################################
# classifier.py
####################################################################
# Contains xgboost. Can process training and testing sets.
####################################################################

import xgboost
import numpy as np
import optuna
import sklearn.metrics

# class Classifier:
#     def __init__(self, estimators, depth, lr):
#         self.bst = XGBClassifier(n_estimators=estimators, max_depth=depth, learning_rate=lr, objective='binary:logistic')
    
#     def train(self, X_train, y_train):
#         self.bst.fit(X_train, y_train)

#     def test(self, X_test, y_test):
#         preds = self.bst.predict(X_test)
#         acc = np.equal(y_test, preds)
#         return preds, acc


def objective(trial, X_train, y_train, X_valid, y_valid):
    # (data, target) = sklearn.datasets.load_breast_cancer(return_X_y=True)
    # train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)
    # dtrain = xgb.DMatrix(train_x, label=train_y)
    # dvalid = xgb.DMatrix(valid_x, label=valid_y)

    

    param = {
        'max_depth': trial.suggest_int('max_depth', 2, 15),
        'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 1.0, 0.05),
        'n_estimators': trial.suggest_int('n_estimators', 1000, 10000, 100),
        # 'eta': trial.suggest_discrete_uniform('eta', 0.01, 0.1, 0.01),
        'reg_alpha': trial.suggest_int('reg_alpha', 1, 50),
        'reg_lambda': trial.suggest_int('reg_lambda', 5, 100),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 20),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
    }

    # if param["booster"] in ["gbtree", "dart"]:
    #     # maximum depth of the tree, signifies complexity of the tree.
    #     param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
    #     # minimum child weight, larger the term more conservative the tree.
    #     param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
    #     param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
    #     # defines how selective algorithm is.
    #     param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
    #     param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    # if param["booster"] == "dart":
    #     param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
    #     param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
    #     param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
    #     param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
    print(X_train)
    print(y_train)
    model = xgboost.XGBClassifier(**param)  
    bst = model.fit(X_train, y_train)
    preds = bst.predict(X_valid)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(y_valid, pred_labels)
    return accuracy




class Classifier:
    def __init__(self, estimators, depth, lr):
        self.bst = xgboost.XGBClassifier(n_estimators=estimators, max_depth=depth, learning_rate=lr, objective='binary:logistic')
    
    def train(self, X_train, y_train, X_valid, X_test, y_valid, y_test):

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_valid, y_valid), n_trials=100, timeout=600)

        self.bst.fit(X_train, y_train)

    def test(self, X_test, y_test):
        preds = self.bst.predict(X_test)
        acc = np.equal(y_test, preds)
        return preds, acc
