# -*- coding: utf-8 -*-
import xgboost as xgb
import math
from util.MongoDB_connection import MongodbUtils
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
import itertools
import pandas as pd
import numpy as np
import copy
from module.modeling.model_general_func import log_loss
from util.decorator import get_func_time
from config import config
from module.modeling.model_xgboost import XGBoostModel
from tomorrow import threads



class MixedModel(object):

    def __init__(self, train_X, train_y, loss_func, param=None, valid_X=None, valid_y=None):
        self.train_X = train_X
        self.train_y = train_y
        self.valid_X = valid_X
        self.valid_y = valid_y
        self.model = None
        self.loss_func = loss_func
        self.param = param
        self.result = {"model": "MixedModel"}
        self.model_pool = []
        self.weights = []

    @get_func_time
    # @threads(2)
    def fit_one_model(self, param):
        _model = param.get("model")
        _param = param.get("param")

        model_input = dict(
            train_X=self.train_X,
            train_y=self.train_y,
            valid_X=self.valid_X,
            valid_y=self.valid_y,
            param=_param,
            loss_func=self.loss_func,
        )

        _mod = _model(**model_input)
        _mod.fit(valid=False)

        return _mod

    def fit_all_models(self, param={}, save=True, valid=True):
        _param = param if param else self.param
        for _p in _param:
            _model = self.fit_one_model(_p)
            self.model_pool.append(copy.copy(_model))

        return None

    def get_weights(self):

        pass

    def regression_weights(self):

        _X = self.predict_all_models(self.train_X)
        _y = self.train_y

        lr = LogisticRegression(C=1., solver='lbfgs')
        lr.fit(_X, _y)

        self.weight_model = lr
        return lr



    def ave_weights(self):
        n_models = len(self.model_pool)
        _w = 1 / float(n_models)
        weight = [_w for i in range(n_models)]
        return weight

    @get_func_time
    def fit(self, param={}, save=True, valid=True):
        _param = param if param else self.param
        for _p in self.param:
            _model = self.fit_one_model(_p)
            self.model_pool.append(copy.copy(_model))

        self.regression_weights()
        # if valid:
        #     _score = self.validate(self.train_X, self.train_y)
        #     if save:
        #         self.result["param"] = _param
        #         self.result["training_score"] = _score
        #         self.result["train_shape"] = self.train_X.shape
        # return _score

    def validate(self, X, y):
        pred = self.predict(X)
        _score = self.loss_func(y, pred)
        return _score

    def predict_one_model(self, X=None, model=None):
        pred = model.predict(X)
        return pred

    # @get_func_time
    def predict_all_models(self, X):
        preds = np.array([])
        for m in self.model_pool:
            _pred = np.array(m.predict(X))
            if len(preds) > 0:
                preds = np.concatenate((preds, np.asmatrix(_pred).T), axis=1)
            else:
                preds = np.asmatrix(_pred).T

        # weighted_pred = [0 for i in range(len(preds[0]))]
        # for p,w in zip(preds,self.weights):
        #     weighted_pred += p * w
        return preds

    def predict(self, X):
        preds = self.predict_all_models(X)
        weighted_preds = self.weight_model.predict_proba(preds)[:, 1]
        return weighted_preds


if __name__ == "__main__":
    # n = 1000
    # tr_X = train[:n]
    # tr_y = target[:n]
    # te_X = train[1001:1100]
    # te_y = target[1001:1100]
    #

    import pickle
    import os
    print os.getcwd()
    print os.chdir("../")
    from module.modeling.model_ensemble import RandomForest, ExtraTrees, GradientBoosting
    # with open("./data/datasets_dev.pkl", "wb") as f:
    #     pickle.dump((tr_X, tr_y, te_X, te_y), f)

    with open("../data/datasets_dev.pkl", "r") as f:
        tr_X, tr_y, te_X, te_y = pickle.load(f)

    NumberInt = lambda x: x
    ObjectId = lambda x: x

    param_0 = {
        "index" : NumberInt(0),
        "colsample_bytree" : 0.7,
        "silent" : NumberInt(1),
        "fold_idx" : NumberInt(1),
        "min_child_weight" : NumberInt(50),
        "subsample" : 0.6,
        "eta" : 0.1,
        "ntree" : NumberInt(75),
        "objective" : "binary:logistic",
        "max_depth" : NumberInt(8)
    }

    rnd = 57
    param_1 = dict(bootstrap=False, class_weight='auto', criterion='entropy',
                    max_depth=None, max_features='sqrt', max_leaf_nodes=None,
                    min_samples_leaf=1, min_samples_split=4,
                    min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
                    oob_score=False, random_state=rnd, verbose=0,
                    warm_start=False)

    param_2 = dict(bootstrap=False, class_weight=None, criterion='entropy',
                      max_depth=None, max_features='sqrt', max_leaf_nodes=None,
                      min_samples_leaf=1, min_samples_split=3,
                      min_weight_fraction_leaf=1e-5, n_estimators=500, n_jobs=-1,
                      oob_score=False, random_state=rnd, verbose=0, warm_start=False)

    param_3 = dict(init=None, learning_rate=0.1, loss='deviance',
                    max_depth=2, max_features=None, max_leaf_nodes=None,
                    min_samples_leaf=1, min_samples_split=3,
                    min_weight_fraction_leaf=0.0, n_estimators=50,
                    presort='auto', random_state=rnd, subsample=1.0, verbose=0,
                    warm_start=False)



    _param_1 = dict(model=XGBoostModel, param=param_0)
    _param_2 = dict(model=RandomForest, param=param_1)
    _param_3 = dict(model=ExtraTrees, param=param_2)
    _param_4 = dict(model=GradientBoosting, param=param_3)

    pipline = [_param_1, _param_2, _param_3, _param_4]


    model_input = dict(
        train_X=tr_X,
        train_y=tr_y,
        valid_X=te_X,
        valid_y=te_y,
        param=pipline,
        loss_func=log_loss,
    )

    xgb_model = MixedModel(**model_input)
    xgb_model.fit()
    xgb_model.regression_weights()
    # xgb_model.weights = xgb_model.ave_weights()
    # xgb_model.predict(xgb_model.valid_X)
    print xgb_model.validate(xgb_model.valid_X, xgb_model.valid_y)


