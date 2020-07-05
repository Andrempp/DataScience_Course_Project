import pandas as pd
from sklearn.naive_bayes import GaussianNB, BernoulliNB

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
selects = [1, 0.9,0.75, 0.6, 0.5]
classifiers = [GaussianNB(), BernoulliNB()]
algs = ["selectkbest", "PCA"]
plt.figure()
fig, axs = plt.subplots(4, len(classifiers), figsize=(16, 14), squeeze=False)
for i in range(len(algs)):
    alg = algs[i]
    for k in range(len(classifiers)):
        f = classifiers[k]
        values = {}
        svalues = {}
        for d in selects:
            yvalues = []
            syvalues = []
            for tr in thresholds:
                datared = datapp.preprocess_alt(data, "class", red_corr=True, tr=tr, n=5, normalization=normalization,
                                  ignore_classes=categoric, as_df=True)
                df = datapp.feature_reduction(datared, "class",["class","id"], d, alg=alg)

                acc, sens,x = eval.train_predict_kfold(df, "class", f, bal=bal)

                yvalues.append(acc)
                syvalues.append(sens)
            values[d] = yvalues
            svalues[d] = syvalues
        plot.multiple_line_chart(axs[0 + 2 * i, k], thresholds, values, '{} with {}'.format(f, str(alg)),
                                 'threshold of reduction', 'accuracy', percentage=False)
        plot.multiple_line_chart(axs[1 + 2 * i, k], thresholds, svalues, '{} with {}'.format(f, str(alg)),
                                 'threshold of reduction', 'sensitivity', percentage=False)

plt.show()
#%%
tr=0.9
f= "PCA"
selectk = 1
datared = datapp.preprocess_alt(data, "class", red_corr=True, tr=tr, n=5, normalization=normalization,
                              ignore_classes=categoric, as_df=True)
df = datapp.feature_reduction(datared, "class",["class","id"], n_features=selectk, alg=f)
df.shape

# %%
#BERNOULLI É PARA DATA DISCRETA ,NAP USAMOS

smoothing = [1e-08, 1e-09, 1e-10]

plt.figure()
fig, axs = plt.subplots(1,2, figsize=(12, 8), squeeze=False)

values = {}
svalues = {}
yvalues = []
syvalues = []
for s in smoothing:
    nb = GaussianNB(var_smoothing=s)
    acc, sens, x = eval.train_predict_kfold(df, "class", nb, bal=bal)
    yvalues.append(acc)
    syvalues.append(sens)
values["NB"] = yvalues
svalues["NB"] = syvalues
plot.multiple_line_chart(axs[0, 0], smoothing, values, 'Naive bayes with changing smoothing',
                         'smoothing value', 'accuracy')
plot.multiple_line_chart(axs[0, 1], smoothing, svalues, 'Naive bayes with changing smoothing',
                         'smoothing value', 'sensitivity')

plt.show()

#smoothing value não faz diferença

#%%
tr=0.9
f= "PCA"
selectk = 1
datared = datapp.preprocess_alt(data, "class", red_corr=True, tr=tr, n=5, normalization=normalization,
                              ignore_classes=categoric, as_df=True)
df = datapp.feature_reduction(datared, "class",["class","id"], n_features=selectk, alg=f)
nb = GaussianNB()

acc, sens,x,  std_acc, std_sens= eval.train_predict_kfold(df, "class", nb, bal=bal, std=True)
print("Accuracy: {} with Standard Deviation: {}\nSensitivity: {} with Standard Deviation: {}"
      .format(acc,std_acc,sens, std_sens))