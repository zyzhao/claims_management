# -*- coding: utf-8 -*-
import xgboost as xgb
import math
from util.MongoDB_connection import MongodbUtils
from sklearn import cross_validation
import itertools
import pandas as pd
import numpy as np
from module.modeling.model_general_func import log_loss
from util.decorator import get_func_time
from config import config

class PresentValidation(object):

    def __init__(self):
        pass


    def show_scores(self, table, query, projection, sort=[]):
        with MongodbUtils(config.IP, config.PORT,
                            config.COLLECTION, table) as connect_db:
            if projection:
                _curcor = connect_db.find(query, fields=projection)
            else:
                _curcor = connect_db.find(query)
        _res = []
        for r in _curcor:
            obs = {}
            for k in projection.keys():
                _li = k.split(".")
                v = r
                for sub_k in _li:
                    v = v.get(sub_k)
                obs[k] = v
            _res.append(obs)


        _res = pd.DataFrame(_res)
        if sort:
            _res.sort(sort, inplace=True)
        print _res

if __name__ == "__main__":
    pv = PresentValidation()
    _proj = {
        "param.ntree":1 ,
        # "param.max_depth":1 ,"param.colsample_bytree":1 ,
        "param.subsample": 1, "param.eta":1,"param.max_depth":1 ,
        "training_score":1, "valid_score":1
    }

    _filter = {"param.ntree":{"$gt": 900}}
    pv.show_scores(config.XGB_MODELING_TABLE, _filter, _proj, "valid_score")







