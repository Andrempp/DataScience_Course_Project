import pandas as pd
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier

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
#%%
loss="deviance"
#%%
min_samples_leaf = [.05, .025, .01, .005, .0025, .001]
n_estimators = [100, 200, 300]
lr = [0.1, 0.3, 0.5, 0.7]

plt.figure()
fig, axs = plt.subplots(2, len(lr), figsize=(12, 7), squeeze=False)
for k in range(len(lr)):
    f = lr[k]
    values = {}
    for d in n_estimators:
        yvalues = []
        for n in min_samples_leaf:
            gb = GradientBoostingClassifier(min_samples_leaf=n, n_estimators=d, loss=loss, learning_rate=f,
                                            random_state=rs)
            gb.fit(trnX, trnY)
            pred = gb.predict(valX)
            yvalues.append(metrics.accuracy_score(valY, pred))
        values[d] = yvalues
    plot.multiple_line_chart(axs[0, k], min_samples_leaf, values, 'Gradient Boosting with %s lf' % f,
                             'min_samples_leaf', 'accuracy')

plt.show()

#%%
lr=0.5
#%%
min_samples_leaf = [.05, .025, .01, .005, .0025, .001]
n_estimators = [100, 200, 300]
max_depth = [3, 10, 15, 25]

plt.figure()
fig, axs = plt.subplots(2, len(max_depth), figsize=(12, 7), squeeze=False)
for k in range(len(max_depth)):
    f = max_depth[k]
    values = {}
    for d in n_estimators:
        yvalues = []
        for n in min_samples_leaf:
            gb = GradientBoostingClassifier(min_samples_leaf=n, n_estimators=d, loss=loss, learning_rate=lr,
                                            random_state=rs, max_depth=f)
            gb.fit(trnX, trnY)
            pred = gb.predict(valX)
            yvalues.append(metrics.accuracy_score(valY, pred))
        values[d] = yvalues
    plot.multiple_line_chart(axs[0, k], min_samples_leaf, values, 'Gradient Boosting with %s lf' % f,
                             'min_samples_leaf', 'accuracy')

plt.show()
#%%
loss = "deviance"
lr = 0.3
min_samples_leaf = 0.001
n_estimators = 300
max_depth = 25

gb = GradientBoostingClassifier(min_samples_leaf=min_samples_leaf, n_estimators=n_estimators, loss=loss,
                                learning_rate=lr,random_state=rs, max_depth=max_depth)
gb.fit(trnX, trnY)
pred = gb.predict(tstX)
print("accuracy: ", metrics.accuracy_score(tstY, pred))
