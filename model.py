import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.model_selection import KFold

class hi_model():
    def __init__(self, num_splits, x_train, y_train, x_test, y_test):
        self.kf = None
        self.num_splits = num_splits
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def run_model(self):
        self.kf = KFold(n_splits=self.num_splits, shuffle=True, random_state=2018)
        train = np.zeros((self.x_train.shape[0],1))
        test = np.zeros((self.x_test.shape[0],1))
        test_pre = np.empty((self.num_splits,self.x_test.shape[0],1))
        cv_scores=[]
        for i,(train_train_index,train_test_index) in enumerate(self.kf.split(self.x_train)):
            tr_x = self.x_train[train_train_index]
            tr_y = self.y_train[train_train_index]
            te_x = self.x_train[train_test_index]
            te_y = self.y_train[train_test_index]

            train_matrix = lgb.Dataset(tr_x, label=tr_y)
            test_matrix = lgb.Dataset(te_x, label=te_y)

            params = {
                      'boosting_type': 'gbdt',
                      'objective': 'binary',
                      'metric': 'auc',
                      'num_leaves': 2**5,
                      'subsample': 0.70,
                      'learning_rate': 0.01,
                      'seed': 2018,
                      'nthread': 12,
                      }

            num_round = 15000
            early_stopping_rounds = 100
            model = lgb.train(params, train_matrix,num_round,valid_sets=test_matrix,
                                early_stopping_rounds=early_stopping_rounds)
            pre= model.predict(te_x,num_iteration=model.best_iteration).reshape((te_x.shape[0],1))
            train[train_test_index]=pre
            test_pre[i, :]= model.predict(self.x_test, num_iteration=model.best_iteration).reshape((self.x_test.shape[0],1))
            cv_scores.append(roc_auc_score(te_y, pre))
        test[:]=test_pre.mean(axis=0)
        self.y_test["orderType"] = test.reshape((-1, 1))