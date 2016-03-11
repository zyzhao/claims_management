# -*- coding: utf-8 -*-
import xgboost as xgb
import math
from util.MongoDB_connection import MongodbUtils
from sklearn import cross_validation
import itertools
import pandas as pd
import numpy as np


class Prediction(object):

    def __init__(self, train, target, test, model, param):
        self.train = train
        self.target = target
        self.test = test
        self.model = model
        self.param = param
        self._mod = None

    def predict(self):
        model_input = dict(
            train_X=self.train,
            train_y=self.target,
            valid_X=self.test,
            param=self.param,
            loss_func=None,
        )
        self._mod = self.model(**model_input)
        self._mod.fit(save=False, valid=False)
        pred = self._mod.predict(self.test)
        return pred

    def make_submission(self, pred, template_dir="", out_dir=""):
        submission = pd.read_csv(template_dir)
        submission.PredictedProb = pred
        submission.to_csv(out_dir, index=False)
        print "Done"

















