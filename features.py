import pandas as pd
import time
import numpy as np
import featuretools as ft
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit
import subprocess
import re
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import math

profile_train=pd.read_csv("./input/userProfile_train.csv")
profile_test=pd.read_csv("./input/userProfile_test.csv")
profile=pd.concat([profile_train,profile_test])

comment_train=pd.read_csv("./input/userComment_train.csv")
comment_test=pd.read_csv("./input/userComment_test.csv")
comment=pd.concat([comment_train,comment_test])

history_train=pd.read_csv("./input/orderHistory_train.csv")
history_test=pd.read_csv("./input/orderHistory_test.csv")
history=pd.concat([history_train,history_test])

action_train=pd.read_csv("./input/action_train.csv")
action_test=pd.read_csv("./input/action_test.csv")
action=pd.concat([action_train,action_test])

future_train=pd.read_csv("./input/orderFuture_train.csv")
future_test=pd.read_csv("./input/orderFuture_test.csv")
future_test['orderType']=-1
future=pd.concat([future_train,future_test])

feats_path = "./output/feats.csv"

def get_date(timestamp) :
    time_local = time.localtime(timestamp)
    dt = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
    return dt

def get_vector(x, model):
    num = len(x)
    vector = np.zeros(50).reshape((1, 50))
    for word in x:
        vector += model[word].reshape((1, 50))
    if num != 0:
        vector /= num
    return vector

def df2ffm(df, fp, target=0):
    #Convert pandas.DataFrame to data format that libffm can directly use
    with open(fp, "w") as f:
        f.truncate()
    field = df.columns.values.tolist()
    feature_dict={}
    del field[target]
    for f in field:
        feature_dict[f] = df[f].unique().tolist()
    # print ("field = ", field)
    # print ("feature_dict = ", feature_dict)
    with open(fp, 'w') as f:
        f.truncate()
        for row in df.values:
            line = str(int(row[0]))
            for index in range(0, len(row)-1):
                feature_index = 0
                for i in range(index+1):
                    if i < index:
                        feature_index += len(feature_dict[field[i]])
                    elif i == index:
                        feature_index += feature_dict[field[i]].index(row[index+1])
                line += " %d:%d:%d" % (index,feature_index,row[index+1])
            line += '\n'
            f.write(line)

def auto_feats_process(feature):
    feature.loc[:,["gender","province","age", "MODE(userComment.orderHistory.city)", "MODE(userComment.orderHistory.country)", "MODE(userComment.orderHistory.continent)"]] \
        = feature.loc[:,["gender","province","age","MODE(userComment.orderHistory.city)", "MODE(userComment.orderHistory.country)", "MODE(userComment.orderHistory.continent)"]].fillna("empty")
    le = LabelEncoder()
    feature.loc[:,["gender"]] = le.fit_transform(feature.loc[:,["gender"]])
    feature.loc[:,["province"]] = le.fit_transform(feature.loc[:,["province"]])
    feature.loc[:,["age"]] = le.fit_transform(feature.loc[:,["age"]])
    feature.loc[:, ["MODE(userComment.orderHistory.city)"]] = le.fit_transform(feature.loc[:, ["MODE(userComment.orderHistory.city)"]])
    feature.loc[:,["MODE(userComment.orderHistory.country)"]] = le.fit_transform(feature.loc[:,["MODE(userComment.orderHistory.country)"]])
    feature.loc[:, ["MODE(userComment.orderHistory.continent)"]] = le.fit_transform(feature.loc[:, ["MODE(userComment.orderHistory.continent)"]])
    train = feature.loc[feature['orderType'] != -1].loc[:, ["orderType","gender","province","age","MODE(userComment.orderHistory.city)","MODE(userComment.orderHistory.country)","MODE(userComment.orderHistory.continent)"]]
    ss = ShuffleSplit(n_splits=2, train_size=0.7, random_state = 0)
    for train_index, val_index in ss.split(train):
        train_train = train.iloc[train_index,:]
        train_val = train.iloc[val_index,:]
    df2ffm(train_train, "./libffm_train.txt", 0)
    df2ffm(train_val, "./libffm_val.txt", 0)
    # transfer sparse data to the probability of orderType==1
    cmd = "ffm-train.exe --auto-stop -r 0.1 -t 100 -p ./libffm_val.txt ./libffm_train.txt ffm_model"
    subprocess.call(cmd, shell=True)
    df2ffm(feature.loc[:,["orderType","gender","province","age", "MODE(userComment.orderHistory.city)", "MODE(userComment.orderHistory.country)", "MODE(userComment.orderHistory.continent)"]], "./libffm.txt", 0)
    with open("./libffm.out", "w") as f:
        f.truncate()
    cmd = "ffm-predict.exe ./libffm.txt ffm_model libffm.out"
    subprocess.call(cmd, shell=True)
    feature.drop(["gender", "province", "age", "MODE(userComment.orderHistory.country)", "MODE(userComment.orderHistory.city)",
         "MODE(userComment.orderHistory.continent)"], axis=1, inplace=True)
    with open("./libffm.out", "r") as f:
        ffm_data = np.array(f.read().splitlines()).reshape((-1, 1))
    feature.loc[:, "ffm_data"] = ffm_data
    # Sentiment analysis with user comments.
    feature.loc[:, "has_order_history"] = feature.loc[:, "MODE(userComment.orderHistory.userid)"].map(
        lambda x: 1 if x is not np.nan else 0)
    feature.loc[:, "has_comment"] = feature.loc[:, "MODE(userComment.orderid)"].map(
        lambda x: 1 if x is not np.nan else 0)
    feature.drop(["MODE(userComment.orderHistory.userid)", "MODE(userComment.orderid)"], axis=1, inplace=True)
    feature.loc[:, "sentiments"] = feature.loc[:, "MODE(userComment.tags)"].fillna("") + \
                                   feature.loc[:,"MODE(userComment.commentsKeyWords)"].fillna("")
    feature.drop(["MODE(userComment.tags)", "MODE(userComment.commentsKeyWords)"], axis=1, inplace=True)
    regEx = re.compile(r'\W+')
    feature.loc[:, "sentiments"] = feature.loc[:, "sentiments"].map(lambda x: regEx.split(x))
    feature.loc[:, "sentiments"] = feature.loc[:, "sentiments"].map(lambda x: list(filter(None, x)) if x is not "" else "空白")
    model_1 = Word2Vec(feature.loc[:, "sentiments"], size=50, min_count=1)
    feature.loc[:, "sentiments"] = feature.loc[:, "sentiments"].map(lambda x: get_vector(x, model=model_1))
    model_2 = KMeans(n_clusters=2, init='k-means++', n_init=5)
    mm = feature.loc[:, "sentiments"].values
    mm = np.vstack(mm)
    feature.loc[:, "sentiments"] = model_2.fit_predict(mm)
    return feature

def gen_auto_feats():
    action["userid"] = action["userid"].astype("str")
    future["userid"] = future["userid"].astype("str")
    history["orderTime"] = history["orderTime"].apply(lambda x:get_date(int(x)))
    history["userid"] = history["userid"].astype("str")
    history["orderid"] = history["orderid"].astype("str")
    history["orderType"] = history["orderType"].astype("str")
    comment["userid"] = comment["userid"].astype("str")
    comment["orderid"] = comment["orderid"].astype("str")
    profile["userid"] = profile["userid"].astype("str")

    es = ft.EntitySet(id="train")
    es = es.entity_from_dataframe(entity_id="userProfile", dataframe=profile, index="userid")
    es = es.entity_from_dataframe(entity_id="userComment", dataframe=comment, index="userid")
    es = es.entity_from_dataframe(entity_id="orderHistory", dataframe=history, index="orderid", time_index="orderTime",
                                  variable_types={"orderType" : ft.variable_types.Categorical})

    relationship_1 = ft.Relationship(es["userProfile"]["userid"], es["userComment"]["userid"])
    es = es.add_relationship(relationship_1)
    relationship_2 = ft.Relationship(es["orderHistory"]["orderid"], es["userComment"]["orderid"])
    es = es.add_relationship(relationship_2)

    feature, _ = ft.dfs(entityset=es, target_entity="userProfile")
    feature = feature.T.drop_duplicates(keep='first').T.reset_index()
    feature = pd.merge(future, feature, how='left', on="userid")
    feature = auto_feats_process(feature)
    feature.fillna(0)
    return feature

def gen_action_feats():
    action_data = action.sort_values(["userid", "actionTime"])
    action_data["actionTime_gap"] = action_data["actionTime"] - action_data["actionTime"].shift(1)
    action_data["actionTime_long"] = action_data["actionTime"].shift(-1) - action_data["actionTime"]
    action_data["date"] = pd.to_datetime(action_data["actionTime"].apply(get_date))
    action_data["weekday"] = action_data["date"].dt.weekday
    action_data["hour"] = action_data["date"].dt.hour
    action_data["month"] = action_data["date"].dt.month
    action_data["day"] = action_data["date"].dt.day
    action_data["year"] = action_data["date"].dt.year
    action_data["minute"] = action_data["date"].dt.minute
    action_data["second"] = action_data["date"].dt.second
    action_last = pd.DataFrame(action_data.groupby(["userid"]).actionTime.max()).reset_index()
    action_last.columns = ["userid", "actionTime_last"]
    action_data = action_data.merge(action_last, on="userid", how="left")
    action_data["actionTime_last_dif"] = action_data["actionTime_last"] - action_data["actionTime"]
    action_567 = action_data[(action_data.actionType >= 5) & (action_data.actionType <= 7)]
    stat_data = pd.DataFrame(profile["userid"])

    action_type = action_data[action_data.actionType == 1]
    action_type_table = action_type.pivot_table(index=["userid"], columns=["year", "month"], values=["day"],
                                                    aggfunc=lambda x: x.count(), fill_value=0)
    action_type_table.columns = action_type_table.columns.map(lambda x: "count_action_type1_"+'_'.join(str(i) for i in x))
    action_type_table.reset_index(inplace=True)
    stat_data = pd.merge(stat_data, action_type_table, how='left', on='userid').fillna(0)

    action_type_table = action_type.pivot_table(index=["userid"], columns=["year", "month", "day"], values=["actionTime"],
                                                    aggfunc=lambda x: x.iloc[-1] - x.iloc[0], fill_value=0)
    action_type_table.columns = action_type_table.columns.map(lambda x: "time_gap_action_type1_"+'_'.join(str(i) for i in x))
    action_type_table.reset_index(inplace=True)
    stat_data = pd.merge(stat_data, action_type_table, how='left', on='userid').fillna(0)

    action_read_table = action_data[action_data.actionType.isin([2, 3, 4])]
    action_read_table = action_read_table.pivot_table(index=["userid", "year", "month", "day"],
                                                    values=["hour", "minute", "second"],
                                                    aggfunc=lambda x: x.iloc[-1] - x.iloc[0], fill_value=0)
    action_read_table["action_read"] = action_read_table["hour"] * 60 * 60 + action_read_table["minute"] * 60 + action_read_table["second"]
    action_read_table = action_read_table["action_read"].reset_index()
    action_read_table = action_read_table[["userid", "action_read"]].groupby("userid")["action_read"].agg(["mean", "std", "max", "min", "median"]).fillna(0)
    action_read_table.reset_index(inplace=True)
    action_read_table.columns = ["userid", "action_read_mean", "action_read_std", "action_read_max", "action_read_min", "action_read_median"]
    stat_data = pd.merge(stat_data, action_read_table, how='left', on='userid').fillna(0)

    for i in [600, 1800, 3600, 36000, 100000, 100000000]:
        action_select = action_567[action_567.actionTime_last_dif < i].copy()
        a = pd.get_dummies(action_select, columns=['actionType']).groupby("userid").sum()
        a = a[[i for i in a.columns if 'actionType' in i]].reset_index()
        stat_data = stat_data.merge(a, on="userid", how='left')

    for i in range(5, 8):
        action_select = action_567[action_567.actionType == i]
        action_long_duration = action_select.groupby("userid").actionTime_long.agg(["mean", "std", "max", "min", "median"])
        action_long_duration = pd.DataFrame(action_long_duration).reset_index()
        action_long_duration.columns = ["userid", "action_type%s_long_mean" % i, "action_type%s_long_std" % i, "action_type%s_long_max" % i,
                                   "action_type%s_long_min" % i, "action_type%s_long_median" % i]
        stat_data = pd.merge(stat_data, action_long_duration, how='left', on='userid').fillna(0)

        action_gap_duration = action_select.groupby("userid").actionTime_gap.agg(["mean", "std", "max", "min", "median"])
        action_gap_duration = pd.DataFrame(action_gap_duration).reset_index()
        action_gap_duration.columns = ["userid", "action_type%s_gap_mean" % i, "action_type%s_gap_std" % i, "action_type%s_gap_max" % i,
                                   "action_type%s_gap_min" % i, "action_type%s_gap_median" % i]
        stat_data = pd.merge(stat_data, action_gap_duration, how='left', on='userid').fillna(0)

        action_duration = action_select.groupby("userid").actionTime.agg(["max", "min"])
        action_duration = pd.DataFrame(action_duration).reset_index()
        action_duration.columns = ["userid", "action_type%s_Time_max" % i, "action_type%s_Time_min" % i]
        stat_data = pd.merge(stat_data, action_duration, how='left', on='userid').fillna(0)

    for a, b in [(6, 5), (7, 5)]:
        stat_data["max_%s-max_%s" % (a, b)] = stat_data["action_type%s_Time_max" % a] - stat_data["action_type%s_Time_max" % b]
        stat_data["min_%s-min_%s" % (a, b)] = stat_data["action_type%s_Time_min" % a] - stat_data["action_type%s_Time_min" % b]
        stat_data["%s_%s_rt" % (a, b)] = stat_data["max_%s-max_%s" % (a, b)] / stat_data["min_%s-min_%s" % (a, b)]
        stat_data["%s_%s_dif" % (a, b)] = stat_data["max_%s-max_%s" % (a, b)] - stat_data["min_%s-min_%s" % (a, b)]

    type_prob = pd.DataFrame(action_data.groupby(["userid", "actionType"]).actionTime.count()).reset_index()
    type_prob.columns = ["userid", "actionType", "type_count"]
    all_count = pd.DataFrame(action_data.groupby("userid").actionType.count()).reset_index()
    all_count.columns = ["userid", "all_count"]
    type_prob = pd.merge(type_prob, all_count, how="left", on="userid").fillna(0)
    type_prob["type_rt"] = type_prob["type_count"] / type_prob["all_count"]
    action_user_type = pd.pivot_table(type_prob, index=["userid"], columns=["actionType"], values="type_rt",
                                      fill_value=0).reset_index()
    stat_data = stat_data.merge(action_user_type, on="userid", how="left")

    action_last = pd.DataFrame(action_data.groupby(["userid", "actionType"]).actionTime.max()).reset_index()
    action_last.columns = ["userid", "actionType", "type_actionTime_last"]
    action_last["actionType"] = action_last["actionType"].apply(lambda x: "action_last_" + str(x))
    max_action_time = pd.DataFrame(action_data.groupby("userid").actionTime.max()).reset_index()
    max_action_time.columns = ["userid", "user_last_time"]
    action_last = pd.merge(action_last, max_action_time, on="userid", how="left").fillna(0)
    action_last["before_type_time_gap"] = action_last["user_last_time"] - action_last["type_actionTime_last"]
    action_user_type = pd.pivot_table(action_last, index=["userid"], columns=["actionType"],
                                      values="before_type_time_gap", fill_value=100000).reset_index()
    stat_data = stat_data.merge(action_user_type, on="userid", how="left")

    stat_data["action_last_5_6"] = stat_data["action_last_5"] - stat_data["action_last_6"]
    stat_data["action_last_1_5"] = stat_data["action_last_1"] - stat_data["action_last_5"]
    stat_data["action_last_1_7"] = stat_data["action_last_1"] - stat_data["action_last_7"]

    action_56 = action_data[(action_data.actionType >= 1) & (action_data.actionType <= 6)]
    action_56 = action_56.sort_values("actionTime")
    action_56 = action_56.drop_duplicates(["userid", "actionType"], keep="last")
    action_56["actionType"] = action_56["actionType"].apply(lambda x: "action_long_" + str(x))
    action_user_type = pd.pivot_table(action_56, index=["userid"], columns=["actionType"],
                                      values="actionTime_long", fill_value=100000).reset_index()
    stat_data = stat_data.merge(action_user_type, on="userid", how="left")
    action_56["actionType"] = action_56["actionType"].apply(lambda x: x.replace("action_long_", "action_gap_"))
    action_user_type = pd.pivot_table(action_56, index=["userid"], columns=["actionType"],
                                      values="actionTime_gap", fill_value=100000).reset_index()
    stat_data = stat_data.merge(action_user_type, on="userid", how="left").fillna(0)
    stat_data["action_long_6_rt"] = stat_data["action_long_6"] / stat_data["action_type6_long_mean"]

    action_time_tmp = pd.DataFrame(action_data.groupby("userid").actionTime.agg(["max", "min", "count"])).reset_index()
    action_time_tmp.columns = ["userid", "last_time", "first_time", "time_count"]
    stat_data = pd.merge(stat_data, action_time_tmp, on="userid", how="left").fillna(0)
    stat_data["userid"] = stat_data["userid"].astype("str")
    return stat_data

def gen_feats():
    action_feats = gen_action_feats()
    auto_feats = gen_auto_feats()
    feats_data = pd.merge(action_feats, auto_feats, how='left', on='userid').fillna(0)
    feats_data.to_csv(feats_path)
    return feats_data
