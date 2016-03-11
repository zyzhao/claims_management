# -*- coding: utf-8 -*-

# X, Y, Xtarget, Ytarget


## XGB
import xgboost as xgb
import math

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
model = xgb.train(final_param, dtrain, 100)
print "end"


model.get_fscore()

train_preds = model.predict(dtrain, ntree_limit=model.best_iteration)

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


test_preds = model.predict(dtest, ntree_limit=model.best_iteration)

log_loss(Xtarget,train_preds)
log_loss(Ytarget,test_preds)





_dtrain = xgb.DMatrix(train, label=target)
_dtest = xgb.DMatrix(test)
print "start"
_model = xgb.train(final_param, _dtrain, 100)
print "end"

log_loss(target,model.predict(_dtrain, ntree_limit=_model.best_iteration))



_test_preds = model.predict(_dtest, ntree_limit=_model.best_iteration)

submission = pd.read_csv('input/sample_submission.csv')
submission.PredictedProb = _test_preds
submission.to_csv('output/simpleblend.csv', index=False)
print "Done"





train.shape
test.shape

import itertools
list(itertools.product([1, 2, 3], [1, 2, 3], [1, 2, 3]))


