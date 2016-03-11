# -*- coding: utf-8 -*-
import os
import pickle
from module.predicting.predict_test import Prediction
from module.modeling.model_xgboost import XGBoostModel
from module.modeling.model_ensemble import RandomForest, ExtraTrees, GradientBoosting
from module.modeling.model_mix import MixedModel

print "Loading"
# os.chdir("../")
print os.getcwd()
with open("data/datasets.pkl", "r") as f:
    res = pickle.load(f)
print "Data loaded"

train = res["train"]
target = res["target"]
test = res["test"]

NumberInt = lambda x: x

#
# param = {
#     "index" : NumberInt(0),
#     "colsample_bytree" : 0.7,
#     "silent" : NumberInt(1),
#     "fold_idx" : NumberInt(1),
#     "min_child_weight" : NumberInt(50),
#     "subsample" : 0.6,
#     "eta" : 0.01,
#     "ntree" : NumberInt(800),
#     "objective" : "binary:logistic",
#     "max_depth" : NumberInt(8)
# }


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

pre = Prediction(model=MixedModel, param=pipline, **res)
_pred = pre.predict()
print _pred

pre.make_submission(_pred, "input/sample_submission.csv", "output/simpleblend.csv")
print "Done"




