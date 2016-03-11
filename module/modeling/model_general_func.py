# -*- coding: utf-8 -*-
import math


def log_loss(y, y_pred, silent=0):
    _target_pred = zip(y, y_pred)
    _cumsum = 0
    # print y
    # print y_pred
    for _p in _target_pred:
        y_i = _p[0]
        p_i = _p[1]
        p_i = p_i if p_i < 1 else p_i - 1e-6
        p_i = p_i if p_i > 0 else p_i + 1e-6
        # print y_i, "/", p_i
        _cumsum += (y_i * (math.log(p_i)) + (1 - y_i) * math.log(1 - p_i))
    score = - _cumsum / len(_target_pred)
    if silent:
        print "score: {score}".format(score=score)
    return score




