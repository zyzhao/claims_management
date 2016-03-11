# -*- coding: utf-8 -*-
import xgboost as xgb
import math
from util.MongoDB_connection import MongodbUtils
from sklearn import cross_validation
import itertools
import pandas as pd
import numpy as np
from grid_search import log_loss

# if "train" not in dir() or "test" not in dir():
#     print "Prepare datasets."
#     from prepare_data import train, test, target
# else:
#     print "Data already exist"


final_param = dict(
    max_depth=6,
    eta=0.055,
    min_child_weight=50,
    objective='binary:logistic',
    subsample=0.6,
    colsample_bytree=0.7,
    silent=0
)


dtrain = xgb.DMatrix(X, label=Xtarget)
dtest = xgb.DMatrix(Y)
print "start"
model = xgb.train(final_param, dtrain, 50)
print "end"


train_preds = model.predict(dtrain, ntree_limit=model.best_iteration)
test_preds = model.predict(dtest, ntree_limit=model.best_iteration)

tr_score_0 = log_loss(Xtarget,train_preds)
te_score_0 = log_loss(Ytarget,test_preds)


X_1 = np.concatenate((X,np.asmatrix(train_preds).T), axis=1)
Y_1 = np.concatenate((Y,np.asmatrix(test_preds).T), axis=1)

# add pred
dtrain_1 = xgb.DMatrix(X_1, label=Xtarget)
dtest_1 = xgb.DMatrix(Y_1)
print "start"
model_1 = xgb.train(final_param, dtrain_1, 50)
print "end"


train_preds_1 = model_1.predict(dtrain_1, ntree_limit=model_1.best_iteration)
test_preds_1 = model_1.predict(dtest_1, ntree_limit=model_1.best_iteration)

tr_score_1 = log_loss(Xtarget,train_preds_1)
te_score_1 = log_loss(Ytarget,test_preds_1)

print  [[tr_score_0, te_score_0],[tr_score_1, te_score_1]]


train_preds.shape
X.shape
np.asmatrix(train_preds).T.shape
np.vstack((X,train_preds))





