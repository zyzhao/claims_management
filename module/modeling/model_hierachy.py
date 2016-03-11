# -*- coding: utf-8 -*-
import numpy as np

MODEL_MAP = {"XGB": XGBoostModel}

class HierachicalModel(object):

    def __init__(self, train_X, train_y, loss_func, param=None, valid_X=None, valid_y=None):
        self.train_X = train_X
        self.train_y = train_y
        self.valid_X = valid_X
        self.valid_y = valid_y
        self.model = None
        self.loss_func = loss_func
        self.param = param
        self.result = {"model": "XGBoostModel"}

    def fit_one_model(self, param):
        print param
        model_class = param.get("model")
        _p = param.get("param")

        model_input = dict(
            train_X=self.train_X,
            train_y=self.train_y,
            valid_X=self.valid_X,
            valid_y=self.valid_y,
            param=_p,
            loss_func=self.loss_func,
        )

        _mod = model_class(**model_input)
        _mod.fit()
        _pred_train = _mod.predict(self.train_X)
        _pred_test = _mod.predict(self.valid_X)
        self.model = _mod

        res = dict(tr_y_pred=_pred_train, te_y_pred=_pred_test)
        return res

    def fit(self,  param={}, save=True, valid=True):
        _param = param if param else self.param
        self.param = _param
        n = len(_param)
        for idx in range(n):
            _p = _param[idx]
            _preds = self.fit_one_model(_p)
            print self.model.validate()

            if idx != (n-1):
                self.train_X = np.concatenate((self.train_X, np.asmatrix(_preds["tr_y_pred"]).T), axis=1)
                self.valid_X = np.concatenate((self.valid_X, np.asmatrix(_preds["te_y_pred"]).T), axis=1)

        _score = None
        if valid:
            _pred = self.model.predict(self.train_X)
            _score = self.loss_func(self.train_y, _pred)
            if save:
                self.result["param"] = _param
                self.result["training_score"] = _score
                self.result["train_shape"] = self.train_X.shape
        return _score

    def validate(self, X=None, y=None, save=True):
        _valid_X = X if X else self.valid_X
        _valid_y = y if y else self.valid_y
        _pred = self.model.predict(_valid_X)
        _score = self.loss_func(_valid_y, _pred)

        if save:
            self.result["valid_score"] = _score
            self.result["valid_shape"] = _valid_X.shape
        return _score


if __name__ == "__main__":
    from module.modeling.model_general_func import log_loss
    from module.modeling.model_xgboost import XGBoostModel


    n = 114321 / 3 * 2
    tr_X = train[:n]
    tr_y = target[:n]
    te_X = train[n:]
    te_y = target[n:]

    NumberInt = lambda x: x
    ObjectId = lambda x: x



    final_param = {
        "index" : NumberInt(0),
        "colsample_bytree" : 0.7,
        "silent" : NumberInt(1),
        "fold_idx" : NumberInt(2),
        "min_child_weight" : NumberInt(50),
        "subsample" : 0.6,
        "eta" : 0.1,
        "ntree" : NumberInt(50),
        "objective" : "binary:logistic",
        "max_depth" : NumberInt(8)
    }

    pipline = [
        {"model": XGBoostModel, "param": final_param},
        {"model": XGBoostModel, "param": final_param},
        {"model": XGBoostModel, "param": final_param}
    ]

    model_input = dict(
        train_X=tr_X,
        train_y=tr_y,
        valid_X=te_X,
        valid_y=te_y,
        param=pipline,
        loss_func=log_loss,
    )

    xgb_model = HierachicalModel(**model_input)
    xgb_model.fit()
    print xgb_model.validate()
    # s = xgb_model.fit_validate(param=final_param)
    # xgb_model.save_result()
    # print s






