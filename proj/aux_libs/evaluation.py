#!/usr/bin/env python
# coding: utf-8
import statistics

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from proj.aux_libs import datapreprocessing as datapp


def general_eval(df, to_clf, col_describe=()):
    data = df.copy()
    columns = data.columns
    data_types = data.dtypes.value_counts()
    col_nulls = data.columns[data.isna().any()].tolist()
    counts = data[to_clf].dropna().value_counts(normalize=True)
    print("\n Total of columns: ", len(columns))
    print("\n Total of rows: ", data.shape[0])
    print("\n Data types:\n", data_types)
    print("\n Columns with null values:", col_nulls)
    print("\n Percentage of classes:\n", counts)
    if len(col_describe) > 0:
        print("Full describe:\n", data[col_describe].describe())



def outliers_category(df, columns, ratio=1.5, by="column"):
    data = df.copy()
    q1 = data[columns].quantile(q=0.25)
    q3 = data[columns].quantile(q=0.75)
    iqr = q3-q1
    lower = q1 - (ratio*iqr)
    upper = q3 + (ratio*iqr)
    if by=="column":
        nout = []
        for i,c in enumerate(columns):
            v = data.loc[ (data[c]>lower[i]) & (data[c]<upper[i])]
            nout.append(len(v))
        return nout
    elif by == "row":
        nout = []
        for index, row in data.iterrows():
            #r_upper = upper - row
            serie1 = row[row.gt(upper)]
            serie2 = row[row.lt(lower)]
            nout.append(len(serie1) + len(serie2))
        return nout
    else:
        raise ValueError("Argument by must be 'row' or 'column'")


def sensitivity(tstY, prdY, labels):
    cfm = metrics.confusion_matrix(tstY, prdY, labels)
    return cfm[0,0]/(cfm[0,0]+cfm[0,1])



#Fazer media dos 3
#Garantir que caiem todos os ids no mesmo fold

def train_predict_kfold(df, to_clf, classifier, k=2, bal=None, std=False,random_state=42):
    data = df.copy()
    columns = data.columns
    y: np.ndarray = data[to_clf].values
    data_X = data.drop(to_clf, axis=1)
    id_index = data_X.columns.get_loc("id")
    X: np.ndarray = data_X.values
    labels = pd.unique(y)

    meaned = datapp.mean_df(data, "id")
    yid: np.ndarray = meaned[to_clf].values.reshape((-1, 1))
    Xid: np.ndarray = meaned.drop(to_clf, axis=1).values

    accuracys = []
    sensitivities = []
    cfms = []
    #max_id = data["id"].max()
    skf = StratifiedKFold(n_splits=k)
    for train_index, test_index in skf.split(Xid, yid):
        func_train = np.vectorize(lambda t: t in train_index)
        func_test = np.vectorize(lambda t: t in test_index)
        ind_train, ind_test = func_train(X[:, id_index]), func_test(X[:, id_index])
        X_train, X_test = X[ind_train], X[ind_test]
        y_train, y_test = y[ind_train], y[ind_test]

        if bal =="smote":
            smote = SMOTE(sampling_strategy='minority', random_state=random_state)
            X_train, y_train = smote.fit_sample(X_train, y_train)
        elif bal == "oversample":
            np_train = np.concatenate((X_train, y_train.reshape((-1,1))), axis=1)
            df_train = pd.DataFrame(np_train, columns=columns)
            df_over = datapp.oversample(df_train, to_clf)
            y_train: np.ndarray = df_over[to_clf].values
            X_train: np.ndarray = df_over.drop(to_clf, axis=1).values

        classifier.fit(X_train, y_train)
        prdY = classifier.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, prdY)
        sens = sensitivity(y_test, prdY, labels)
        cfm = metrics.confusion_matrix(y_test, prdY, labels)
        accuracys.append(accuracy)
        sensitivities.append(sens)
        cfms.append(cfm.astype("int"))
    acc = round(sum(accuracys)/len(accuracys), 3)
    sen = round(sum(sensitivities)/len(sensitivities), 3)
    cfm = sum(cfms)

    if std:
        stdacc = round(statistics.stdev(accuracys), 3)
        stdsen = round(statistics.stdev(sensitivities), 3)
        return (acc, sen, cfm, stdacc, stdsen)
    return (acc, sen, cfm)


#def predict_unsupervised(df, to_clf, random_state):

def measures_cluster(pred, true, labels):
    pred2 = [0 if i==1 else 1 for i in pred]

    acc1 = metrics.accuracy_score(true, pred)
    acc2 = metrics.accuracy_score(true, pred2)
    acc = max(acc1, acc2)

    sens1 = sensitivity(true, pred, labels)
    sens2 = sensitivity(true, pred2, labels)

    sens = max(sens1, sens2)

    return (acc, sens)





def cut(df, bins, ig_classes, cut="cut"):
    labels = list(map(str,range(1,bins+1)))
    dfc = df.copy()
    if cut == "cut":
        for col in dfc:
            if col not in ig_classes:
                dfc[col] = pd.cut(dfc[col],bins,labels=labels)
    elif cut == "qcut":
         for col in dfc:
            if col not in ig_classes:
                dfc[col] = pd.qcut(dfc[col],bins,labels=labels)
    return dfc

def dummy(df,ig_classes):
    dfc = df.copy()
    dummylist = []
    for att in dfc:
        if att in ig_classes: dfc[att] = dfc[att].astype('category')
        dummylist.append(pd.get_dummies(dfc[[att]]))
    dummified_df = pd.concat(dummylist, axis=1)
    return dummified_df

from mlxtend.frequent_patterns import apriori, association_rules #for ARM

def freq_itemsets(df, minpaterns=30):
    dfc = df.copy()
    frequent_itemsets = {}
    minsup = 1.0
    while minsup>0:
        minsup = minsup*0.9
        frequent_itemsets = apriori(dfc, min_support=minsup, use_colnames=True)
        if len(frequent_itemsets) >= minpaterns:
            print("\nMinimum support:",minsup)
            break
    print("Number of found patterns:",len(frequent_itemsets))
    return frequent_itemsets


def assoc_rules(fi, orderby="lift", inverse=False, min_confidence=0.7):
    rules = association_rules(fi, metric="confidence", min_threshold=min_confidence)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules = rules[(rules['antecedent_len'] >= 2) & (rules['confidence'] > min_confidence)]
    if orderby != None:
        rules = rules.sort_values(by=[orderby], ascending=inverse)
    return rules


def conf_matrix():
    cfm = metrics.confusion_matrix(d[3], prdY, d[4])