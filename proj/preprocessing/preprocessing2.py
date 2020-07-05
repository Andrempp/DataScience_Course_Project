import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

from proj.aux_libs import datapreprocessing as datapp, evaluation as eval, plot
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

#%%
"""
per_class = [-1, 2000, 1000, 500]
classifiers = [GaussianNB(), DecisionTreeClassifier(), RandomForestClassifier()]

plt.figure()
fig, axs = plt.subplots(1, len(per_class), figsize=(15,10), squeeze=False)
for n in range(len(per_class)):
    print("per class cycle")
    xvalues = []
    yvalues = []
    num = per_class[n]
    for clf in classifiers:
        print("classifier cycle")
        datasub = datapp.subsample(data, to_clf, per_class=num)
        y: np.ndarray = datasub[to_clf].values
        y = y.astype('int')
        X: np.ndarray = datasub.drop(to_clf, axis=1).values
        trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=train_ratio, stratify=y, random_state=rs)
        clf.fit(trnX, trnY)
        pred = clf.predict(tstX)
        xvalues.append(metrics.accuracy_score(tstY, pred))
        yvalues.append(str(clf.__class__))
    plot.bar_chart(axs[0, n], yvalues, xvalues, 'Histogram with {} per class'.format(str(num)), str(num),'accuracy'
                   ,percentage=True)
fig.tight_layout()
plt.show()
"""

#Redução piora um bocado
#%%
#df = datapp.subsample(data, to_clf)

trnX, tstX, valX, trnY, tstY, valY = datapp.subsample_split(data, to_clf, categoric, "standard")

print(trnX.shape)
print(tstX.shape)
print(valX.shape)
#%%
outliers_col = eval.outliers_category(data, data.columns.tolist(), ratio=1.5, by="column")
outliers_row = eval.outliers_category(data, data.columns.tolist(), ratio=1.5, by="row")
plt.figure()
bin = 20
plt.hist(outliers_col, bin)
plt.title = "Frequencies of quantity of outliers (by column)"
plt.show()
plt.figure()
bin = 20
plt.hist(outliers_row, bin,cumulative=False)
plt.title = "Frequencies of quantity of outliers (by row)"
plt.show()

#%%
norms = ["standard", "minmax", None]
classifiers = [BernoulliNB(), DecisionTreeClassifier(), RandomForestClassifier()]
names = ["BernouliNB", "DecisionTree", "RandomForest"]

plt.figure()
fig, axs = plt.subplots(1, len(norms), figsize=(15,10), squeeze=False)

for n in range(len(norms)):
    xvalues = []
    yvalues = []
    norm = norms[n]
    for clf in classifiers:
        trnX, tstX, valX, trnY, tstY, valY = datapp.subsample_split(data, to_clf, categoric, norm)
        clf.fit(trnX, trnY.astype('int'))
        pred = clf.predict(tstX)
        xvalues.append(metrics.accuracy_score(tstY, pred))
        yvalues.append(str(clf.__class__))
    plot.bar_chart(axs[0, n], names, xvalues, 'Histogram for {}'.format(str(norm)), str(norm), 'accuracy', percentage=True)
fig.tight_layout()
plt.show()

#COMO A MAIOR PARTE É CATEOGIRCA, E ESTAS NÃO SAO NORMALIZADAS, A DIFERENÇA É MINIMA, MESMO ASSIM VAMOS COM STANDARD
#%%
normalization = "standard"
#%%
