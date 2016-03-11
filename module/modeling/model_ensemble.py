# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from util.decorator import get_func_time


class EnsembleModel(object):
    def __init__(self, train_X, train_y, loss_func, param=None, valid_X=None, valid_y=None):
        self.train_X = train_X
        self.train_y = train_y
        self.valid_X = valid_X
        self.valid_y = valid_y
        self.model = None
        self.loss_func = loss_func
        self.param = param
        # self.result = {"model": "XGBoostModel"}
        pass

    @get_func_time
    def fit(self, param={}, save=True, valid=True):
        _param = param if param else self.param
        self.param = _param
        self.model.fit(self.train_X, self.train_y)

        _score = None
        if valid:
            _score = self.validate(self.train_X, self.train_y)
            if save:
                self.result["param"] = _param
                self.result["training_score"] = _score
                self.result["train_shape"] = self.train_X.shape
        return _score

    def validate(self, X, y):
        pred = self.predict(X)
        _score = self.loss_func(y, pred)
        return _score

    def predict(self, X=None):
        pred = self.model.predict_proba(X)[:, 1]
        return pred


class RandomForest(EnsembleModel):
    def __init__(self, **kwargs):
        super(RandomForest, self).__init__(**kwargs)
        self.model = RandomForestClassifier(**self.param)
        self.result = {"model": "RandomForest"}


class ExtraTrees(EnsembleModel):
    def __init__(self, **kwargs):
        super(ExtraTrees, self).__init__(**kwargs)
        self.model = ExtraTreesClassifier(**self.param)
        self.result = {"model": "ExtraTrees"}


class GradientBoosting(EnsembleModel):
    def __init__(self, **kwargs):
        super(GradientBoosting, self).__init__(**kwargs)
        self.model = GradientBoostingClassifier(**self.param)
        self.result = {"model": "GradientBoosting"}


if __name__ == "__main__":
    from module.modeling.model_general_func import log_loss

    n = 1000
    tr_X = train[:n]
    tr_y = target[:n]
    te_X = train[1001:1100]
    te_y = target[1001:1100]

    NumberInt = lambda x: x
    ObjectId = lambda x: x

    # final_param = dict(
    #     bootstrap=True, class_weight='auto', criterion='entropy',
    #     max_depth=None, max_features='sqrt', max_leaf_nodes=None,
    #     min_samples_leaf=1, min_samples_split=4,
    #     min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
    #     oob_score=False, verbose=0,
    #     warm_start=False,
    # )
    final_param = dict(
        init=None, learning_rate=0.1, loss='deviance',
        max_depth=2, max_features=None, max_leaf_nodes=None,
        min_samples_leaf=1, min_samples_split=3,
        min_weight_fraction_leaf=0.0, n_estimators=50,
        presort='auto', random_state=1, subsample=1.0, verbose=0,
        warm_start=False
    )

    model_input = dict(
        train_X=tr_X,
        train_y=tr_y,
        valid_X=te_X,
        valid_y=te_y,
        param=final_param,
        loss_func=log_loss,
    )

    xgb_model = GradientBoosting(**model_input)
    print xgb_model.fit(valid=True)
    # print xgb_model.param
    # xgb_model.fit(param=final_param)
    # xgb_model.validate()
    # s = xgb_model.fit_validate(param=final_param)
    # xgb_model.save_result()
