import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
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

#%%
datared = datapp.preprocess_alt(data, "class", red_corr=True, tr=0.9, n=5, normalization="standard",
                              ignore_classes=categoric, as_df=True)

#%%
#id pode ajudar na classificação no conjunto de teste, para evitar, retiramos esse atributo.
#to_remove = ["id"]
#datam = data.copy()
#data = data.drop(columns=to_remove)
#%%
# Vemos através de vários heatmaps que há algumas variaveis bastante correlacionadas. Vamos pegar nessas variavies,
# e transformá-las, variáveis com correlação superior a tr vão passar a ser representadas pela media, mediana e std dos
# seus valores.

#
# Vamos testar normalização
#%%

norms = ["standard", "minmax", None]
names = ["Gaussian", "DecisionTree", "RandomForest"]
classifiers = [GaussianNB(), DecisionTreeClassifier(), RandomForestClassifier()]

plt.figure()
fig, axs = plt.subplots(1, len(norms), figsize=(10,10), squeeze=False)

for n in range(len(norms)):
    xvalues = []
    yvalues = []
    norm = norms[n]
    for clf in classifiers:
        datared = datapp.preprocess_alt(data, "class", normalization=norm,ignore_classes=categoric, as_df=True)
        # df = datapp.feature_reduction(datared, "class", ["class", "id"], 1, alg="selectkbest")

        acc, sens, x = eval.train_predict_kfold(datared, "class", clf, bal=None)
        xvalues.append(acc)
        yvalues.append(str(clf.__class__))
    plot.bar_chart(axs[0, n], names, xvalues, 'Histogram for {}'.format(str(norm)), str(norm), 'accuracy', percentage=False)
fig.tight_layout()
plt.show()

#%%
normalization = "standard"

#%%
#VAMOS TESTAR BALANCEAMENTO
balances = ["smote", "oversample", None]
classifiers = [GaussianNB(), BernoulliNB(), DecisionTreeClassifier()]
names = ["Gaussian", "Bernoulli", "DecisionTree"]

plt.figure()
fig, axs = plt.subplots(1, len(balances), figsize=(10,10), squeeze=False)

for n in range(len(balances)):
    xvalues = []
    yvalues = []
    balance = balances[n]
    for clf in classifiers:
        datared = datapp.preprocess_alt(data, "class", normalization=normalization,
                                    ignore_classes=categoric, as_df=True)
        #df = datapp.feature_reduction(datared, "class", ["class", "id"], 1, alg="selectkbest")

        acc, sens, x = eval.train_predict_kfold(datared, "class", clf, bal=balance)

        xvalues.append(acc)
        yvalues.append(str(clf.__class__))
    plot.bar_chart(axs[0, n], names, xvalues, 'Histogram for {}'.format(str(balance)), str(balance), 'accuracy',
                   percentage=False)
fig.tight_layout()
plt.show()

#%%
bal = "smote"
normalization = "standard"

#não se remove outliers

#%%
