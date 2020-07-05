import pandas as pd
from sklearn.tree import DecisionTreeClassifier

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

# %%
#Testar reduction de features
#Acrescentar SMOTE
thresholds = [1 ,0.95, 0.90, 0.8]
selects = [1, 0.9,0.75, 0.6, 0.5]
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

            tree = DecisionTreeClassifier(random_state=rs)
            acc, sens, _ = eval.train_predict_kfold(df, "class", tree, bal=bal)

            yvalues.append(acc)
            syvalues.append(sens)
        values[d] = yvalues
        svalues[d] = syvalues
    plot.multiple_line_chart(axs[0, k], thresholds, values, 'Decision Trees with %s reduction' % f,
                             'threshold of reduction', 'accuracy')
    plot.multiple_line_chart(axs[1, k], thresholds, svalues, 'Decision Trees with %s reduction' % f,
                             'threshold of reduction', 'sensitivity', percentage=False)

plt.show()
#%%
tr=0.95
f= "selectkbest"
selectk = 1
datared = datapp.preprocess_alt(data, "class", red_corr=True, tr=tr, n=5, normalization=normalization,
                              ignore_classes=categoric, as_df=True)
df = datapp.feature_reduction(datared, "class",["class","id"], n_features=selectk, alg=f)
df.shape

# %%
# PARAMETERS PARA TESTAR: criterion, splitter, max_depth, min_sample_leaf, min_samples_split

min_samples_leaf = [.05, .025, .01, .0075, .005, .0025, .001]
max_depths = [5, 10, 25, 50]
criteria = ['entropy', 'gini']

plt.figure()
fig, axs = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
for k in range(len(criteria)):
    f = criteria[k]
    values = {}
    svalues = {}
    for d in max_depths:
        yvalues = []
        syvalues = []
        for n in min_samples_leaf:
            tree = DecisionTreeClassifier(min_samples_leaf=n, max_depth=d, criterion=f, random_state=rs)
            acc, sens, _ = eval.train_predict_kfold(df, "class", tree, bal=bal)
            yvalues.append(acc)
            syvalues.append(sens)
        values[d] = yvalues
        svalues[d] = syvalues
    plot.multiple_line_chart(axs[0, k], min_samples_leaf, values, 'Decision Trees with %s criteria' % f,
                             'min_samples_leaf', 'accuracy')
    plot.multiple_line_chart(axs[1, k], min_samples_leaf, svalues, 'Decision Trees with %s criteria' % f,
                             'min_samples_leaf', 'sensitivity', percentage=False)

plt.show()

# %%
criterion = "entropy"
# %%
# max_depth é exatamente igual e, 25 e 50, porque chega a um ponto em que pára (ver a fundo)

min_samples_leaf = [.05, .025, .01, .0075, .005, .0025, .001]
max_depths = [5, 10, 25]
splitters = ['best', 'random']

plt.figure()
fig, axs = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
for k in range(len(splitters)):
    f = splitters[k]
    values = {}
    svalues = {}
    for d in max_depths:
        yvalues = []
        syvalues = []
        for n in min_samples_leaf:
            tree = DecisionTreeClassifier(min_samples_leaf=n, max_depth=d, criterion=criterion, random_state=rs,
                                          splitter=f)
            acc, sens, _ = eval.train_predict_kfold(df, "class", tree, bal=bal)
            yvalues.append(acc)
            syvalues.append(sens)
        values[d] = yvalues
        svalues[d] = syvalues
    plot.multiple_line_chart(axs[0, k], min_samples_leaf, values, 'Decision Trees with %s splitter' % f,
                             'min_samples_leaf', 'accuracy')
    plot.multiple_line_chart(axs[1, k], min_samples_leaf, svalues, 'Decision Trees with %s splitter' % f,
                             'min_samples_leaf', 'sensitivity', percentage=False)

plt.show()
# %%
splitter = "best"
# %%
min_samples_leaf = [.01, .0075, .005, .0025, .001]
max_depths = [5, 10, 25]
min_samples_split = [2, 3, 4]

plt.figure()
fig, axs = plt.subplots(2, 3, figsize=(15, 4), squeeze=False)
for k in range(len(min_samples_split)):
    f = min_samples_split[k]
    values = {}
    svalues = {}
    for d in max_depths:
        yvalues = []
        syvalues = []
        for n in min_samples_leaf:
            tree = DecisionTreeClassifier(min_samples_leaf=n, max_depth=d, criterion=criterion, random_state=rs,
                                          splitter=splitter, min_samples_split=f)
            acc, sens, _ = eval.train_predict_kfold(df, "class", tree, bal=bal)
            yvalues.append(acc)
            syvalues.append(sens)
        values[d] = yvalues
        svalues[d] = syvalues
    plot.multiple_line_chart(axs[0, k], min_samples_leaf, values, 'Decision Trees with %s min_samples_split' % f,
                             'min_samples_leaf', 'accuracy')
    plot.multiple_line_chart(axs[1, k], min_samples_leaf, svalues, 'Decision Trees with %s min_samples_split' % f,
                             'min_samples_leaf', 'sensitivity', percentage=False)

plt.show()

# 10 E 25 SOBREPOSTO
# MIN SAMPLES SPLIT INDIFERENTE
# %%
tr=0.95
f= "selectkbest"
selectk = 1
criterion = "entropy"
splitter = "best"
max_depth = 25
min_samples_leaf = .005

datared = datapp.preprocess_alt(data, "class", red_corr=True, tr=tr, n=5, normalization=normalization,
                              ignore_classes=categoric, as_df=True)
df = datapp.feature_reduction(datared, "class",["class","id"], n_features=selectk, alg=f)

tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, max_depth=max_depth, criterion=criterion,
                              random_state=rs, splitter=splitter)

acc, sens, std_acc, std_sens= eval.train_predict_kfold(df, "class", tree, bal=bal, std=True)
print("Accuracy: {} with Standard Deviation: {}\nSensitivity: {} with Standard Deviation: {}"
      .format(acc,std_acc,sens, std_sens))
