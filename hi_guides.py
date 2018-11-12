from features import gen_feats, feats_path
from model import hi_model
import pandas as pd
import numpy as np
import os

if os.path.exists(feats_path):
    try:
        feature = pd.read_csv(feats_path, index_col=[0])
        if feature.empty:
            feature = gen_feats()
    except:
        feature = gen_feats()
else:
    feature = gen_feats()
feats = feature.columns.tolist()
feats.remove("userid")
feats.remove("orderType")
y_train = feature[feature.orderType!=-1]['orderType'].reset_index(drop=True)
y_train = y_train.astype(int).values
x_train = feature[feature.orderType!=-1][feats].reset_index(drop=True)
x_test = feature[feature.orderType==-1][feats].reset_index(drop=True)
y_test = feature[feature.orderType==-1][["userid", "orderType"]].reset_index(drop=True)
x_train = np.array(x_train)
x_test = np.array(x_test)
himodel = hi_model(2, x_train, y_train, x_test, y_test)
himodel.run_model()
result = himodel.y_test
result.to_csv("predict.csv", index=False)
