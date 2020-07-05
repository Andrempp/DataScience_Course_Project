import pandas as pd
import time
from sklearn.ensemble import GradientBoostingClassifier

from proj.aux_libs import datapreprocessing as datapp, evaluation as eval, plot
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

pd.set_option("display.max_columns", 15)
rs = 32
data = pd.read_csv('../../data/pd_speech_features.csv', skiprows=[0])
to_clf = "class"
categoric = ["gender", "id"]
to_remove = ["id"]
data.shape

normalization = "standard"
bal = "smote"

#%%
thresholds = [1,0.95, 0.90, 0.8]
selects = [1, 0.9, 0.8, 0.75, 0.6]
algs = ["selectkbest", "PCA"]
plt.figure()
fig, axs = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
for k in range(len(algs)):
    f = algs[k]
    values = {}
    svalues = {}
    for d in selects:
        yvalues = []
        syvalues = []
        for tr in thresholds:
            datared = datapp.preprocess_alt(data, "class", red_corr=True, tr=tr, n=5, normalization=normalization,
                              ignore_classes=categoric, as_df=True)
            df = datapp.feature_reduction(datared, "class",["class","id"], d, alg=f)
            xg_clf = GradientBoostingClassifier()
            startTime = time.process_time()
            acc, sens, _ = eval.train_predict_kfold(df, "class", xg_clf, bal=bal)
            print(time.process_time() - startTime)
            yvalues.append(acc)
            syvalues.append(sens)
        values[d] = yvalues
        svalues[d] = syvalues
    plot.multiple_line_chart(axs[0, k], thresholds, values, 'XGBoost with %s reduction' % f,
                             'threshold of reduction', 'accuracy')
    plot.multiple_line_chart(axs[1, k], thresholds, svalues, 'XGBoost with %s reduction' % f,
                             'threshold of reduction', 'sensitivity', percentage=False)

plt.show()

#%%
tr=0.9
f= "selectkbest"
selectk = 0.75
datared = datapp.preprocess_alt(data, "class", red_corr=True, tr=tr, n=5, normalization=normalization,
                              ignore_classes=categoric, as_df=True)
df = datapp.feature_reduction(datared, "class",["class","id"], n_features=selectk, alg=f)
df.shape

#%%
min_samples_leaf = [.05, .025, .01, .005, .0025, .001]
n_estimators = [100, 200, 300]
loss = ['deviance', 'exponential']

plt.figure()
fig, axs = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
for k in range(len(loss)):
    f = loss[k]
    values = {}
    svalues = {}
    for d in n_estimators:
        yvalues = []
        syvalues = []
        for n in min_samples_leaf:
            gb = GradientBoostingClassifier(min_samples_leaf=n, n_estimators=d, loss=f, random_state=rs)
            acc, sens, _ = eval.train_predict_kfold(df, "class", gb, bal=bal)
            yvalues.append(acc)
            syvalues.append(sens)
        values[d] = yvalues
        svalues[d] = syvalues
    plot.multiple_line_chart(axs[0, k], min_samples_leaf, values, 'Gradient Boosting with %s loss' % f,
                             'min_samples_leaf', 'accuracy')
    plot.multiple_line_chart(axs[1, k], min_samples_leaf, svalues, 'Gradient Boosting with %s loss' % f,
                             'min_samples_leaf', 'sensitivity', percentage=False)

plt.show()


#%%
loss = "exponential"
#%%

min_samples_leaf = [.05, .025, .01, .005, .0025, .001]
n_estimators = [100, 200, 300]
lr = [0.1, 0.3, 0.5, 0.7]


plt.figure()
fig, axs = plt.subplots(2, len(lr), figsize=(12, 7), squeeze=False)
for k in range(len(lr)):
    f = lr[k]
    values = {}
    svalues = {}
    for d in n_estimators:
        yvalues = []
        syvalues = []
        for n in min_samples_leaf:
            gb = GradientBoostingClassifier(min_samples_leaf=n, n_estimators=d, loss=loss, learning_rate=f,
                                            random_state=rs)
            acc, sens, _ = eval.train_predict_kfold(df, "class", gb, bal=bal)
            yvalues.append(acc)
            syvalues.append(sens)
        values[d] = yvalues
        svalues[d] = syvalues
    plot.multiple_line_chart(axs[0, k], min_samples_leaf, values, 'Gradient Boosting with %s lr' % f,
                             'min_samples_leaf', 'accuracy')
    plot.multiple_line_chart(axs[1, k], min_samples_leaf, svalues, 'Gradient Boosting with %s lr' % f,
                             'min_samples_leaf', 'sensitivity', percentage=False)

plt.show()

#%%
lr = 0.3
#%%

min_samples_leaf = [.05, .025, .01, .005, .0025, .001]
n_estimators = [25,50,100]
max_depth = [3, 10, 15, 25]
#aumentar numero de estimators à vontade porque o algoritmo é naturalmente resistente a overfitting
plt.figure()
fig, axs = plt.subplots(2, len(max_depth), figsize=(12, 7), squeeze=False)
for k in range(len(max_depth)):
    f = max_depth[k]
    values = {}
    svalues = {}
    for d in n_estimators:
        yvalues = []
        syvalues = []
        for n in min_samples_leaf:
            gb = GradientBoostingClassifier(min_samples_leaf=n, n_estimators=d, loss=loss, learning_rate=lr,
                                            random_state=rs, max_depth=f)
            acc, sens, _ = eval.train_predict_kfold(df, "class", gb, bal=bal)
            yvalues.append(acc)
            syvalues.append(sens)
        values[d] = yvalues
        svalues[d] = syvalues
    plot.multiple_line_chart(axs[0, k], min_samples_leaf, values, 'Gradient Boosting with %s max_depth' % f,
                             'min_samples_leaf', 'accuracy')
    plot.multiple_line_chart(axs[1, k], min_samples_leaf, svalues, 'Gradient Boosting with %s max_depth' % f,
                             'min_samples_leaf', 'sensitivity', percentage=False)

plt.show()

#%%
tr=0.9
f= "selectkbest"
selectk = 0.9
loss = "exponential"
lr = 0.3
min_samples_leaf = 0.01
n_estimators = 100
max_depth = 3

datared = datapp.preprocess_alt(data, "class", red_corr=True, tr=tr, n=5, normalization=normalization,
                              ignore_classes=categoric, as_df=True)
df = datapp.feature_reduction(datared, "class",["class","id"], n_features=selectk, alg=f)


gb = GradientBoostingClassifier(min_samples_leaf=min_samples_leaf, n_estimators=n_estimators, loss=loss,
                                learning_rate=lr,random_state=rs, max_depth=max_depth)


acc, sens, _ ,std_acc, std_sens= eval.train_predict_kfold(df, "class", gb, bal=bal, std=True)
print("Accuracy: {} with Standard Deviation: {}\nSensitivity: {} with Standard Deviation: {}"
      .format(acc,std_acc,sens, std_sens))

