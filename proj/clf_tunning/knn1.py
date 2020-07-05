from proj.aux_libs import datapreprocessing as datapp, evaluation as eval, plot
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
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

            knn = KNeighborsClassifier()
            acc, sens, x = eval.train_predict_kfold(df, "class", knn, bal=bal)

            yvalues.append(acc)
            syvalues.append(sens)
        values[d] = yvalues
        svalues[d] = syvalues
    plot.multiple_line_chart(axs[0, k], thresholds, values, 'KNN with %s reduction' % f,
                             'threshold of reduction', 'accuracy')
    plot.multiple_line_chart(axs[1, k], thresholds, svalues, 'KNN with %s reduction' % f,
                             'threshold of reduction', 'sensitivity', percentage=False)

plt.show()
#%%
tr=0.9
f= "selectkbest"
selectk = 1
datared = datapp.preprocess_alt(data, "class", red_corr=True, tr=tr, n=5, normalization=normalization,
                              ignore_classes=categoric, as_df=True)
df = datapp.feature_reduction(datared, "class",["class","id"], n_features=selectk, alg=f)
df.shape



#%%

nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
dist = ['manhattan', 'euclidean', 'chebyshev']
weights = ['uniform', 'distance']
values = {}
svalues = {}

plt.figure()
fig, axs = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
for w in range(len(weights)):
    weight = weights[w]
    for d in dist:
        yvalues = []
        syvalues = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d, weights=weight)
            acc, sens, x = eval.train_predict_kfold(df, "class", knn, bal=bal)
            yvalues.append(acc)
            syvalues.append(sens)
        values[d] = yvalues
        svalues[d] = syvalues
    plot.multiple_line_chart(axs[0, w], nvalues, values, 'KNN variants with weight: %s' % weight, 'n', 'accuracy',
                             percentage=False)
    plot.multiple_line_chart(axs[1, w], nvalues, svalues, 'KNN variants with weight: %s' % weight, 'n', 'sensitivity',
                             percentage=False)
plt.show()

#DÁ IGUAL COM OS 2 WEIGHTS

#%%

metric = "manhattan"
n_neighbors=1

knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
acc, sens, x, std_acc,std_sens = eval.train_predict_kfold(df, "class", knn, bal=bal, std=True)
print("Accuracy: {} with Standard Deviation: {}\nSensitivity: {} with Standard Deviation: {}"
      .format(acc,std_acc,sens, std_sens))
