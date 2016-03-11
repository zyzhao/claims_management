# -*- coding: utf-8 -*-
import xgboost as xgb
import math
from util.MongoDB_connection import MongodbUtils
from sklearn import cross_validation
import itertools
import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from module.modeling.model_general_func import log_loss
from module.modeling.model_xgboost import XGBoostModel
from util.decorator import get_func_time
from tomorrow import threads
import time

NODE_LIMIT = 10000
THREADS = 2

class ParameterOptimization(object):
    def __init__(self, train, target, params=[], n_fold=3, save=False):
        self.train = train
        self.target = target
        self.params = params
        self.n_fold = n_fold
        self.KFolder_index = []
        self.param_list = pd.DataFrame()
        self.save = save

    def create_fold_index(self, n_fold=None):
        _n = n_fold if n_fold else self.n_fold
        skf = StratifiedKFold(self.target, n_folds=_n, shuffle=True)

        _idx_list = []
        for train_index, test_index in skf:
            _idx = dict(tr_idx=train_index, te_idx=test_index)
            _idx_list.append(_idx)

        self.KFolder_index = _idx_list

        return _idx_list

    def create_grid_seach_params(self, params={}, shuffle=True):
        _params = params if params else self.params
        param_list = pd.DataFrame()

        tunning_params, fix_params = {}, {}
        n_comb = 1
        for i, j in _params.items():
            if isinstance(j, list):
                if len(j) > 1:
                    tunning_params[i] = j
                    n_comb *= len(j)
                else:
                    fix_params[i] = j[0]
            else:
                fix_params[i] = j

        if n_comb > NODE_LIMIT:
            print "[Warning]: There are too many combination for training!"
        else:
            param_list = pd.DataFrame(list(itertools.product(*tunning_params.values())),
                                      columns=tunning_params.keys())
            for i, j in fix_params.items():
                param_list[i] = j

            print "Shape of param. list (%s, %s)" % param_list.shape

            if shuffle:
                rand_idx = list(np.random.permutation(param_list.index))
                param_list = param_list.iloc[rand_idx].reset_index()
            self.param_list = param_list

        return param_list

    @get_func_time
    def run_grid_search(self, model=None, param_list=None):
        param_list = param_list if param_list else self.param_list
        param_idx_list = param_list.index

        nrow, ncol = param_list.shape

        tmp = []
        st = time.time()
        for idx in param_idx_list:
            obs = param_list.loc[idx].to_dict()
            # Get parameters
            print "[%s/%s]modeling"% (str(idx), str(nrow))
            _res = [self.fit_model_by_folder(_fold_idx, obs, model, st)
                    for _fold_idx in range(len(self.KFolder_index))]
            tmp.append(_res)
            # for _fold in self.KFolder_index:
            #     # print "[%s/%s]Parse parameters" % (str(idx), str(nrow))
            #     tr_idx = _fold["tr_idx"]
            #     te_idx = _fold["te_idx"]
            #
            #     model_input = dict(
            #         train_X=self.train.iloc[tr_idx],
            #         train_y=self.target.iloc[tr_idx],
            #         valid_X=self.train.iloc[te_idx],
            #         valid_y=self.target.iloc[te_idx],
            #         param=obs,
            #         loss_func=log_loss,
            #     )
            #     # Training
            #     print "[%s/%s]modeling"% (str(idx), str(nrow))
            #     print obs
            #     _mod = model(**model_input)
            #     _score = _mod.fit_validate()
            #     print "[%s/%s] Score: %s" % (str(idx), str(nrow), str(_score))
            #     # Make record
            #     # _res.update(obs)
            #     print "----------------------"

            # with MongodbUtils("localhost", 27017, "Kaggle", "claims_management_grid_search") as connect_db:
            #     connect_db.insert(_res)
        return tmp

    @threads(3)
    def fit_model_by_folder(self, folder_idx, obs, model, st=0):
        _fold = self.KFolder_index[folder_idx]
        # print "[%s/%s]Parse parameters" % (str(idx), str(nrow))
        param = obs.copy()
        param["fold_idx"] = folder_idx

        tr_idx = _fold["tr_idx"]
        te_idx = _fold["te_idx"]

        model_input = dict(
            train_X=self.train.iloc[tr_idx],
            train_y=self.target.iloc[tr_idx],
            valid_X=self.train.iloc[te_idx],
            valid_y=self.target.iloc[te_idx],
            param=param,
            loss_func=log_loss,
        )
        # Training
        # print param
        _mod = model(**model_input)
        _res= _mod.fit_validate()
        if _res.get("status", "") == "success" and self.save:
            _mod.save_result(other={"fold_idx": folder_idx})

        print "Time used %s" %  (str(time.time() - st))
        print "----------------------"
        return "Done"


if __name__ == "__main__":

    training_params = dict(
        max_depth=[8],
        eta=[0.005],
        min_child_weight=[50],
        objective='binary:logistic',
        subsample=[0.6],
        colsample_bytree=[0.7],
        silent=1,
        ntree=[50, 75, 100],
    )

    po = ParameterOptimization(train[:100], target[:100], training_params)

    po.create_fold_index()
    print po.create_grid_seach_params().shape
    po.run_grid_search(model=XGBoostModel)
    # po.param_list
    # len(po.KFolder_index[1]["tr_idx"])


# 1-284
# 3-96
# 10-61
# 15-59

