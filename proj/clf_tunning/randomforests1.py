import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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

normalization = "standard"
bal = "smote"
# %%
#Testar reduction de features
#Acrescentar SMOTE
thresholds = [1 ,0.95, 0.90, 0.8]
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
            rf = RandomForestClassifier(random_state=rs)
            acc, sens, _ = eval.train_predict_kfold(df, "class", rf, bal=bal)
            yvalues.append(acc)
            syvalues.append(sens)
        values[d] = yvalues
        svalues[d] = syvalues
    plot.multiple_line_chart(axs[0, k], thresholds, values, 'Random Forests with %s reduction' % f,
                             'threshold of reduction', 'accuracy')
    plot.multiple_line_chart(axs[1, k], thresholds, svalues, 'Random Forests with %s reduction' % f,
                             'threshold of reduction', 'sensitivity', percentage=False)

plt.show()
#%%
tr=0.95
f= "selectkbest"
selectk = 0.6
datared = datapp.preprocess_alt(data, "class", red_corr=True, tr=tr, n=5, normalization=normalization,
                              ignore_classes=categoric, as_df=True)
df = datapp.feature_reduction(datared, "class",["class","id"], n_features=selectk, alg=f)
df.shape

#%%

n_estimators = [50, 100, 200, 300, 400]
max_depths = [5, 10, 25]    # vimos que 25 e 50 sobrepostos
max_features = ['sqrt', 'log2', 0.1, 0.2]
#a = math.ceil(len(max_features) / 2)

plt.figure()
fig, axs = plt.subplots(2, len(max_features), figsize=(20, 8), squeeze=False)
for k in range(len(max_features)):
    print("max feature cycle")
    f = max_features[k]
    values = {}
    svalues = {}
    for d in max_depths:
        print("max depth cycle")
        yvalues = []
        syvalues = []
        for n in n_estimators:
            rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f, random_state=rs)
            #rf = GaussianNB()
            acc, sens, _ = eval.train_predict_kfold(df, "class", rf, bal=bal)
            yvalues.append(acc)
            syvalues.append(sens)
        values[d] = yvalues
        svalues[d] = syvalues
    plot.multiple_line_chart(axs[0, k], n_estimators, values, 'Random Forests with %s features' % f,
                             'nr estimators',
                             'accuracy', percentage=False)
    plot.multiple_line_chart(axs[1, k], n_estimators, svalues, 'Random Forests with %s features' % f,
                             'nr estimators', 'sensitivity', percentage=False)

plt.show()
#%%
max_features="log2"
#%%


n_estimators = [100, 200, 300, 400]
max_depths = [5, 10, 25]
criterions = ['gini', 'entropy']

plt.figure()
fig, axs = plt.subplots(2, len(criterions), figsize=(20, 8), squeeze=False)
for k in range(len(criterions)):
    print("ciclo fora")
    f = criterions[k]
    values = {}
    svalues = {}
    for d in max_depths:
        print("ciclo dentro")
        yvalues = []
        syvalues = []
        for n in n_estimators:
            rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=max_features, criterion=f,
                                        random_state=rs)
            acc, sens, _ = eval.train_predict_kfold(df, "class", rf, bal=bal)
            yvalues.append(acc)
            syvalues.append(sens)
        values[d] = yvalues
        svalues[d] = syvalues
    plot.multiple_line_chart(axs[0, k], n_estimators, values, 'Random Forests with %s criterion' % f,
                             'nr estimators', 'accuracy', percentage=False)
    plot.multiple_line_chart(axs[1, k], n_estimators, svalues, 'Random Forests with %s criterion' % f,
                             'nr estimators', 'sensitivity', percentage=False)

plt.show()

#%%
criterion="entropy"
#%%

n_estimators = [5, 10, 25, 50,100, 200]
min_samples_split = [2, 3, 4]
max_depths = [10, 25]


plt.figure()
fig, axs = plt.subplots(2, len(min_samples_split), figsize=(20, 8), squeeze=False)
values = {}
svalues = {}
for m in range(len(min_samples_split)):
    print("Ciclo fora")
    min_s = min_samples_split[m]
    for d in max_depths:
        print("ciclo dentro")
        yvalues = []
        syvalues = []
        for n in n_estimators:
            rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=max_features,
                                        criterion=criterion, random_state=rs, min_samples_split=min_s)
            acc, sens, _ = eval.train_predict_kfold(df, "class", rf, bal=bal)
            yvalues.append(acc)
            syvalues.append(sens)
        values[d] = yvalues
        svalues[d] = syvalues
    plot.multiple_line_chart(axs[0, m], n_estimators, values, 'Random Forests with %s min_samples split' % min_s,
                         'nr estimators', 'accuracy', percentage=False)
    plot.multiple_line_chart(axs[1, m], n_estimators, svalues, 'Random Forests with %s min_samples split' % min_s,
                             'nr estimators', 'sensitivity', percentage=False)

plt.show()
#%%
min_samples_split=3
n_estimators=25
max_depth=25
#%%
max_features="log2"
min_samples_split=4
n_estimators=50
criterion="entropy"
max_depth=25


rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
                                    criterion=criterion, random_state=rs, min_samples_split= min_samples_split)

acc, sens, std_acc, std_sens= eval.train_predict_kfold(df, "class", rf, bal=bal, std=True)
print("Accuracy: {} with Standard Deviation: {}\nSensitivity: {} with Standard Deviation: {}"
      .format(acc,std_acc,sens, std_sens))

#%%
