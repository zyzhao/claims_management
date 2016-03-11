# -*- coding: utf-8 -*-
import xgboost as xgb
import math
from util.MongoDB_connection import MongodbUtils
from sklearn import cross_validation
import itertools
import pandas as pd
import numpy as np
from grid_search import log_loss

if "train" not in dir() or "test" not in dir():
    print "Prepare datasets."
    from prepare_data import train, test, target
else:
    print "Data already exist"


NumberInt = lambda x: x
ObjectId = lambda x: x

final_param = {
    "_id" : ObjectId("56c4120c37a5b202410ae59f"),
    "colsample_bytree" : 0.7,
    "silent" : NumberInt(1),
    "eta" : 0.05,
    "te_log_loss" : 0.4628132640279909,
    "subsample" : 0.3,
    "seed" : NumberInt(1),
    "min_child_weight" : NumberInt(50),
    "ntree" : NumberInt(150),
    "objective" : "binary:logistic",
    "test_size" : 0.3,
    "max_depth" : NumberInt(9),
    "tr_log_loss" : 0.4605970014422287
}
params_keys = ['subsample', 'eta', 'colsample_bytree', 'silent',
               'objective', 'max_depth', 'min_child_weight']

_param = {k:final_param.get(k) for k in params_keys}
_ntree = final_param.get("ntree", 25)

_dtrain = xgb.DMatrix(train, label=target)
_dtest = xgb.DMatrix(test)
print "start"
_model = xgb.train(_param, _dtrain, _ntree)
print "end"

log_loss(target,_model.predict(_dtrain, ntree_limit=_model.best_iteration))


_test_preds = _model.predict(_dtest, ntree_limit=_model.best_iteration)

submission = pd.read_csv('input/sample_submission.csv')
submission.PredictedProb = _test_preds
submission.to_csv('output/simpleblend.csv', index=False)
print "Done"











