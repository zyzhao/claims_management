# -*- coding: utf-8 -*-
import xgboost as xgb
import math
from util.MongoDB_connection import MongodbUtils
from sklearn import cross_validation
import itertools
import pandas as pd
import numpy as np
from module.modeling.model_general_func import log_loss
from util.decorator import get_func_time
from config import config


class XGBoostModel(object):
    def __init__(self, train_X, train_y, loss_func, param=None, valid_X=None, valid_y=None):
        self.train_X = train_X
        self.train_y = train_y
        self.valid_X = valid_X
        self.valid_y = valid_y
        self.model = None
        self.loss_func = loss_func
        self.param = param
        self.result = {"model": "XGBoostModel"}

    @get_func_time
    def fit(self, param={}, save=True, valid=True):
        _param = param if param else self.param
        self.param = _param
        _ntree = self.param.get("ntree")

        _dtrain = xgb.DMatrix(self.train_X, label=self.train_y)

        _model = xgb.train(_param, _dtrain, _ntree)
        self.model = _model

        _score = None
        if valid:
            _score = self.validate(self.train_X, self.train_y)
            if save:
                self.result["param"] = _param
                self.result["training_score"] = _score
                self.result["train_shape"] = self.train_X.shape
        return _score

    # @get_func_time
    def validate(self, X=None, y=None, save=True):
        # _valid_X = X if X else self.valid_X
        # _valid_y = y if y else self.valid_y
        _pred = self.predict(X)
        _score = self.loss_func(y, _pred)

        if save:
            self.result["valid_score"] = _score
            self.result["valid_shape"] = X.shape
        return _score

    # @get_func_time
    def predict(self, X=None):
        _X = X if str(type(X)) != "<type 'NoneType'>" else self.valid_X
        _dt = xgb.DMatrix(_X)
        _pred = self.model.predict(_dt, ntree_limit=self.model.best_iteration)
        return _pred

    # @get_func_time
    def fit_validate(self, param={}, X=None, y=None):
        res = dict(status="", result={})
        if self.param_check(param):
            print "Pass param_check"
            self.fit(param)
            _score = self.validate(X, y)
            res['status'] = "success"
            res['result'] = {"score": _score}
        else:
            res['status'] = "failed"
        return res

    def param_check(self, param=None):
        _param = param if param else self.param
        _valid = False

        with MongodbUtils(config.IP, config.PORT,
                          config.COLLECTION, config.XGB_MODELING_TABLE) as connect_db:
            _exist = connect_db.find_one({"param": _param})

            if _exist:
                print "Parameter already exist!"
            else:
                _valid = True
        return _valid

    def save_result(self, other={}, check_exist=False):
        _exist = False
        if check_exist:
            with MongodbUtils(config.IP, config.PORT,
                              config.COLLECTION, config.XGB_MODELING_TABLE) as connect_db:
                _exist = connect_db.find_one({"param": self.param})

        if not _exist:
            if other:
                self.result.update(other)
            with MongodbUtils(config.IP, config.PORT,
                              config.COLLECTION, config.XGB_MODELING_TABLE) as connect_db:
                connect_db.insert(self.result)
        return "Done"


if __name__ == "__main__":
    n = 1000
    tr_X = train[:n]
    tr_y = target[:n]
    te_X = train[1001:1100]
    te_y = target[1001:1100]

    NumberInt = lambda x: x
    ObjectId = lambda x: x

    final_param = {
        "colsample_bytree": 0.7,
        "silent": NumberInt(1),
        "eta": 0.05,
        "te_log_loss": 0.4628132640279909,
        "subsample": 0.3,
        "seed": NumberInt(1),
        "min_child_weight": NumberInt(50),
        "ntree": NumberInt(150),
        "objective": "binary:logistic",
        "test_size": 0.3,
        "max_depth": NumberInt(9),
        "tr_log_loss": 0.4605970014422287
    }

    model_input = dict(
        train_X=tr_X,
        train_y=tr_y,
        valid_X=te_X,
        valid_y=te_y,
        param=final_param,
        loss_func=log_loss,
    )

    xgb_model = XGBoostModel(**model_input)
    xgb_model.fit(param=final_param)
    xgb_model.validate()
    s = xgb_model.fit_validate(param=final_param)
    xgb_model.save_result()
    print s
