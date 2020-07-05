import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from proj.aux_libs import datapreprocessing as datapp, plot
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 15)

columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology'
    , 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
           'Horizontal_Distance_To_Fire_Points',
           'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2',
           'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6',
           'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13',
           'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
           'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
           'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
           'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Cover_Type']

categoric = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2',
             'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6',
             'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13',
             'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
             'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
             'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
             'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']
rs = 32
data = pd.read_csv('../../data/covtype.data', names=columns)
to_clf = 'Cover_Type'
train_ratio = 0.7
normalization = "standard"
# %%
trnX, tstX, valX, trnY, tstY, valY = datapp.subsample_split(data, to_clf, categoric, "standard")
trnY = trnY.astype("int")
valY = valY.astype("int")

labels = pd.unique(data[to_clf].values)
#%%
accs=[]
cfms = []
#%%
#NAIVE BAYES

nb = BernoulliNB()

nb.fit(trnX, trnY)
pred = nb.predict(tstX)
acc = metrics.accuracy_score(tstY, pred)
cfm = metrics.confusion_matrix(tstY, pred, labels)

accs.append(acc)
cfms.append(cfm)

print("Naive Bayes")
print("Accuracy: {} \n".format(acc))

#%%
#Decision Tree
criterion = "entropy"
splitter = "best"
max_depth = 25
min_samples_leaf = .005

tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, max_depth=max_depth, criterion=criterion,
                              random_state=rs, splitter=splitter)

tree.fit(trnX, trnY)
pred = tree.predict(tstX)
acc = metrics.accuracy_score(tstY, pred)
cfm = metrics.confusion_matrix(tstY, pred, labels)

accs.append(acc)
cfms.append(cfm)

print("Decision Tree")
print("Accuracy: {} \n".format(acc))

#%%
#RandomForest
max_features="log2"
min_samples_split=4
n_estimators=50
criterion="entropy"
max_depth = 25

rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
                                    criterion=criterion, random_state=rs, min_samples_split= min_samples_split)
rf.fit(trnX, trnY)
pred = rf.predict(tstX)
acc = metrics.accuracy_score(tstY, pred)
cfm = metrics.confusion_matrix(tstY, pred, labels)

accs.append(acc)
cfms.append(cfm)

print("Random Forest")
print("Accuracy: {} \n".format(acc))

#%%
#KNN
metric = "manhattan"
n_neighbors=1

knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
knn.fit(trnX, trnY)

pred = knn.predict(tstX)
acc = metrics.accuracy_score(tstY, pred)
cfm = metrics.confusion_matrix(tstY, pred, labels)

accs.append(acc)
cfms.append(cfm)

print("KNN")
print("Accuracy: {} \n".format(acc))
#%%

gb = GradientBoostingClassifier()
gb.fit(trnX, trnY)

pred = gb.predict(tstX)
acc = metrics.accuracy_score(tstY, pred)
cfm = metrics.confusion_matrix(tstY, pred, labels)

accs.append(acc)
cfms.append(cfm)

print("Gradient Boosting")
print("Accuracy: {} \n".format(acc))

#%%
names = ["NB", "DecisionTree", "RandomForest", "KNN", "GradientBoosting"]


plt.figure()
plt.figure()
plot.bar_chart(plt.gca(), names, accs, 'Comparison of Datasets', '', 'accuracy', percentage=True)
plt.xlabel("Classifiers")
plt.ylim(0, 1.0)
plt.show()

plt.figure()
fig, axs = plt.subplots(1,len(cfms), figsize=(14, 6), squeeze=False)
for m in range(len(cfms)):
    plot.plot_confusion_matrix(axs[0, m], cfms[m].astype("int"), labels, title_m=names[m])
plt.show()



