# -*- coding: utf-8 -*-
import sys
import os
# sys.path.append("../")
import pickle
from module.modeling.param_opt import ParameterOptimization
from module.modeling.model_xgboost import XGBoostModel

if __name__ == "__main__":
    print "Loading"
    # os.chdir("../")
    print os.getcwd()
    with open("data/datasets.pkl", "r") as f:
        res = pickle.load(f)

    train = res["train"]
    target = res["target"]
    test = res["test"]

    print "Start Modeling"
    training_params = dict(
        max_depth=[8],
        eta=[0.1],
        min_child_weight=[1],
        objective='binary:logistic',
        subsample=[0.6],
        colsample_bytree=[0.8],
        silent=1,
        ntree=[1000],
    )
    # training_params = dict(
    #     max_depth=[8],
    #     eta=[0.1],
    #     min_child_weight=[50],
    #     objective='binary:logistic',
    #     subsample=[0.6],
    #     colsample_bytree=[0.7],
    #     silent=1,
    #     ntree=[50],
    # )

    po = ParameterOptimization(train, target, training_params, save=True)
    # po = ParameterOptimization(train[:50], target[:50], training_params, save=False)


    po.create_fold_index()
    print po.create_grid_seach_params().shape
    po.run_grid_search(model=XGBoostModel)



