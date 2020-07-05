import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from proj.aux_libs import datapreprocessing as datapp, plot
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 15)

columns = ['Elevation','Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology'
           ,'Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
            'Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4','Soil_Type1','Soil_Type2','Soil_Type3','Soil_Type4','Soil_Type5','Soil_Type6',
            'Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type10','Soil_Type11', 'Soil_Type12', 'Soil_Type13',
            'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19','Soil_Type20',
            'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
            'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
            'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Cover_Type']

categoric = ['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4','Soil_Type1','Soil_Type2',
             'Soil_Type3','Soil_Type4','Soil_Type5','Soil_Type6',
            'Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type10','Soil_Type11', 'Soil_Type12', 'Soil_Type13',
            'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19','Soil_Type20',
            'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
            'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
            'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']
rs = 32
data = pd.read_csv('../../data/covtype.data', names=columns)
to_clf = 'Cover_Type'
train_ratio = 0.7
normalization = "standard"
#%%
"""
df = datapp.subsample(data, to_clf)
df = datapp.preprocess(df, to_clf, normalization=normalization, ignore_classes=categoric, as_df=True)

y: np.ndarray = data[to_clf].values.astype("int")
X: np.ndarray = data.drop(to_clf, axis=1).values

#trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=train_ratio, stratify=y, random_state=rs)

trnX, tstX, valX, trnY, tstY, valY = eval.tvt_split(df, to_clf)

testeX = np.append(tstX, valX, axis=0)
testeY = np.append(tstY, valY, axis=0)

labels = pd.unique(y)
"""
trnX, tstX, valX, trnY, tstY, valY = datapp.subsample_split(data, to_clf, categoric, "standard")
trnY = trnY.astype("int")
valY = valY.astype("int")

#%%
classifiers = [GaussianNB(), BernoulliNB(), MultinomialNB()]
names = ['Gaussian', 'Bernoulli', 'Multinomial']
plt.figure(figsize=(10, 8))
xvalues = []
yvalues = []
for k in range(len(classifiers)):
    clf = classifiers[k]
    if k == 2:
        a = min( np.min(trnX),  np.min(trnY),  np.min(tstX),  np.min(trnY))
        trnX2 = trnX + abs(a)
        trnY2 = (trnY + abs(a)).astype("int")
        tstY2 = (tstY + abs(a)).astype("int")
        tstX2 = tstX + abs(a)
        clf.fit(trnX2, trnY2)
        pred = clf.predict(tstX2)
        xvalues.append(metrics.accuracy_score(tstY2, pred))
    else:
        clf.fit(trnX, trnY)
        pred = clf.predict(tstX)
        xvalues.append(metrics.accuracy_score(tstY, pred))
    yvalues.append(str(clf.__class__))
plot.bar_chart(plt.gca(), names, xvalues, 'Histogram with various NB', '', 'accuracy'
               , percentage=True)
plt.show()

#BERNOULII É MELHOR, PORUQE BERNOULLI É BOM PARA VARIAVEIS BINARIAS/CATEGORICAS E TEMOS BUES

#%%
alphas = [1, 0.8, 0.6, 0.4, 0.2, 0]

plt.figure()
fig, axs = plt.subplots(1, 1, figsize=(16,10), squeeze=False)
values = {}
yvalues = []
for a in alphas:
    nb = BernoulliNB(alpha=a)
    nb.fit(trnX, trnY)
    pred = nb.predict(valX)
    yvalues.append(metrics.accuracy_score(valY, pred))
values["Bern"] = yvalues
plot.multiple_line_chart(axs[0, 0], alphas, values, 'Bernoulli with various alpha',
                         'nr estimators', 'accuracy', percentage=True)


plt.show()

#Alpha não faz diferença excepto sendo 0 que piora
#%%
#binarize
#%%
binarizes = [1, 0.8, 0.6, 0.4, 0.2, 0]

plt.figure()
fig, axs = plt.subplots(1, 1, figsize=(16, 10), squeeze=False)
values = {}
yvalues = []
for a in binarizes:
    nb = BernoulliNB(binarize=a)
    nb.fit(trnX, trnY)
    pred = nb.predict(valX)
    yvalues.append(metrics.accuracy_score(valY, pred))
values["Bern"] = yvalues
plot.multiple_line_chart(axs[0, 0], binarizes, values, 'Bernoulli with various binarize values',
                         'nr estimators', 'accuracy', percentage=True)

plt.show()

#Binarize não faz muita diferença, deixamos default 0,  excepto no 1 que piora
#%%


nb = BernoulliNB()
nb.fit(trnX, trnY)
pred = nb.predict(tstX)
print("accuracy: ", metrics.accuracy_score(tstY, pred))