import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

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
data = pd.read_csv('proj/data/covtype.data', names=columns)
to_clf = 'Cover_Type'
train_ratio = 0.7
normalization = "standard"

# %%
trnX, tstX, valX, trnY, tstY, valY = datapp.subsample_split(data, to_clf, categoric, "standard")
trnY = trnY.astype("int")
valY = valY.astype("int")

# %%
n_estimators = [50, 100, 200, 300, 400]
max_depths = [5, 10, 25, 50]
max_features = ['sqrt', 'log2', 0.15, 0.3]

plt.figure()
fig, axs = plt.subplots(1, len(max_features), figsize=(18, 5), squeeze=False)
for k in range(len(max_features)):
    print("max feature cycle")
    f = max_features[k]
    values = {}
    for d in max_depths:
        print("max depth cycle")
        yvalues = []
        for n in n_estimators:
            rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f, random_state=rs)
            # rf = GaussianNB()
            rf.fit(trnX, trnY)
            pred = rf.predict(valX)
            yvalues.append(metrics.accuracy_score(valY, pred))
        values[d] = yvalues
    plot.multiple_line_chart(axs[0, k], n_estimators, values, 'Random Forests with %s features' % f,
                             'nr estimators',
                             'accuracy', percentage=False)

plt.show()
# %%

max_features = 0.3

# %%
n_estimators = [50, 100, 200, 300, 400]
max_depths = [10, 25, 50]
criterions = ['gini', 'entropy']

plt.figure()
fig, axs = plt.subplots(1, len(criterions), figsize=(12, 8), squeeze=False)
for k in range(len(criterions)):
    print("max feature cycle")
    f = criterions[k]
    values = {}
    for d in max_depths:
        print("max depth cycle")
        yvalues = []
        for n in n_estimators:
            rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=max_features, criterion=f,
                                        random_state=rs)
            rf.fit(trnX, trnY)
            pred = rf.predict(valX)
            yvalues.append(metrics.accuracy_score(valY, pred))
        values[d] = yvalues
    plot.multiple_line_chart(axs[0, k], n_estimators, values, 'Random Forests with %s features' % f,
                             'nr estimators',
                             'accuracy', percentage=False)
plt.show()
# %%
criterion = "entropy"
# %%
n_estimators = [200, 300, 400, 450, 500, 550]
max_depths = [25, 50]
min_samples_split = [2, 3, 4]

plt.figure()
fig, axs = plt.subplots(1, len(min_samples_split), figsize=(12, 6), squeeze=False)
for k in range(len(min_samples_split)):
    print("max feature cycle")
    f = min_samples_split[k]
    values = {}
    for d in max_depths:
        print("max depth cycle")
        yvalues = []
        for n in n_estimators:
            rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=max_features, criterion=criterion,
                                        random_state=rs)
            rf.fit(trnX, trnY)
            pred = rf.predict(valX)
            yvalues.append(metrics.accuracy_score(valY, pred))
        values[d] = yvalues
    plot.multiple_line_chart(axs[0, k], n_estimators, values, 'Random Forests with %s features' % f,
                             'nr estimators',
                             'accuracy', percentage=False)
plt.show()

# INDIFERENTE
# %%
max_features = 0.3
n_estimators = 550
criterion = "entropy"
max_depth = 25

rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
                            criterion=criterion, random_state=rs)

rf.fit(trnX, trnY)
pred = rf.predict(tstX)
print("accuracy: ", metrics.accuracy_score(tstY, pred))
