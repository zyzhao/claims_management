# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from subprocess import check_output
import random
import sys
import pickle

reload(sys)
sys.setdefaultencoding('utf8')


rnd = 57
maxCategories = 20

class FeatureEngineering():

    def __init__(self, train, test):
        self.train = train
        self.test = test


    def col_process_get_dummies(self, colname):
        _tr_dummies = pd.get_dummies(self.train[colname], prefix=colname,
                                     dummy_na=True).astype('int8')
        _te_dummies = pd.get_dummies(self.test[colname], prefix=colname,
                                     dummy_na=True).astype('int8')
        pass










