# -*- coding: utf-8 -*-
import itertools
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import model_general_func as mgf
import random



class ModelProcessing(object):

    def __init__(self, train, target, test_size=0.3, seed=None):

        if not isinstance(seed,int):
            seed = random.randint(0,10000)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(train, target, test_size=test_size, random_state=seed)

    def pre_process(self):



        pass





# for col in train[:100].columns:
#
# for col in train[:100].columns:
#     print col
#
#     print col
#
# xxx =  pd.read_csv('input/train.csv')
# xxx.shape
# train.shape


