# -*- coding: utf-8 -*-
import xgboost as xgb
import math
from util.MongoDB_connection import MongodbUtils
from sklearn import cross_validation
import itertools
import pandas as pd
import numpy as np


if "train" not in dir() or "test" not in dir():
    print "Prepare datasets."
    from prepare_data import train, test, target
else:
    print "Data already exist"





def log_loss(y,y_pred):
    _target_pred = zip(y, y_pred)
    _cumsum = 0
    for _p in _target_pred:
        y_i = _p[0]
        p_i = _p[1]
        _cumsum += (y_i * (math.log(p_i)) + (1 - y_i) * math.log(1 - p_i))
    score = - _cumsum / len(_target_pred)
    print "score: {score}".format(score=score)
    return score


def training(train, target, param, ntree=20, test_size=0.3, seed=None,
             out_key=["tr_log_loss", "te_log_loss"]):
    score = None
    other_param = dict(test_size=0.2)
    if isinstance(seed, int):
        other_param["random_state"] = seed
    tr_X, te_X, tr_y, te_y = cross_validation.train_test_split(train, target, **other_param)

    dtrain = xgb.DMatrix(tr_X, label=tr_y)
    dtest = xgb.DMatrix(te_X)
    print "Start modeling"
    model = xgb.train(param, dtrain, ntree)
    print "Finish modeling"
    tr_log_loss = log_loss(tr_y,model.predict(dtrain, ntree_limit=model.best_iteration))
    te_log_loss = log_loss(te_y,model.predict(dtest, ntree_limit=model.best_iteration))

    _res = dict(param=param, ntree=ntree, test_size=test_size, seed=seed,
                tr_log_loss=tr_log_loss, te_log_loss=te_log_loss)
    if out_key:
        _res = {k:_res[k] for k in _res if k in _res.keys()}

    return _res


# training_params = dict(
#     max_depth=[3, 6, 9],
#     eta=[0.005, 0.05, 0.01],
#     min_child_weight=[50],
#     objective='binary:logistic',
#     subsample=[0.3, 0.6],
#     colsample_bytree=[0.3, 0.5, 0.7],
#     silent=1,
#     ntree=[25, 75, 150, 200],
#     seed=[1, 100, 200]
# )

training_params = dict(
    max_depth=[8, 11, 15],
    eta=[0.005, 0.01],
    min_child_weight=[50],
    objective='binary:logistic',
    subsample=[0.5],
    colsample_bytree=[0.7],
    silent=1,
    ntree=[300, 500, 800],
    seed=[1, 100, 200]
)
params_keys = ['subsample', 'eta', 'colsample_bytree', 'silent',
               'objective', 'max_depth', 'min_child_weight']


tunning_params, fix_params = {}, {}
for i,j in training_params.items():
    if isinstance(j, list):
        tunning_params[i] = j
    else:
        fix_params[i] = j


param_list = pd.DataFrame(list(itertools.product(*tunning_params.values())),
                          columns=tunning_params.keys())
print param_list.shape


for i, j in fix_params.items():
    param_list[i] = j

nrow, ncol = param_list.shape
rand_idx = list(np.random.permutation(param_list.index))
for idx in rand_idx:
    obs = param_list.loc[idx].to_dict()
    with MongodbUtils("localhost", 27017, "Kaggle", "claims_management_grid_search") as connect_db:
        if_exist = connect_db.find_one(obs)
    if not if_exist:
        # Get parameters
        print "[%s/%s]Parse parameters" % (str(idx), str(nrow))
        _param = {k:obs.get(k) for k in params_keys}
        _other_param = {k:obs.get(k) for k in [key for key in obs.keys() if key not in params_keys]}

        # Training
        print "[%s/%s]modeling"% (str(idx), str(nrow))
        print obs
        _res = training(train, target, _param, **_other_param)

        # Make record
        _res.update(obs)
        # print _res

        with MongodbUtils("localhost", 27017, "Kaggle", "claims_management_grid_search") as connect_db:
            connect_db.insert(_res)
    else:
        print "[%s/%s]param. already exisit!" % (str(idx), str(nrow))


























