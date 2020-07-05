import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

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

print(len(categoric))
rs = 32
data = pd.read_csv('../../data/covtype.data', names=columns)
to_clf = 'Cover_Type'
train_ratio = 0.7
normalization = "standard"
#%%
trnX, tstX, valX, trnY, tstY, valY = datapp.subsample_split(data, to_clf, categoric, "standard")
trnY = trnY.astype("int")
valY = valY.astype("int")

#%%
nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
dist = ['manhattan', 'euclidean', 'chebyshev']
weights = ['uniform', 'distance']
values = {}

plt.figure()
fig, axs = plt.subplots(1, len(weights), figsize=(12, 7), squeeze=False)
for w in range(len(weights)):
    weight = weights[w]
    for d in dist:
        yvalues = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d, weights=weight)
            knn.fit(trnX, trnY)
            pred = knn.predict(valX)
            yvalues.append(metrics.accuracy_score(valY, pred))
        values[d] = yvalues
    plot.multiple_line_chart(axs[0, w], nvalues, values, 'KNN variants with weight: %s' % weight, 'n', 'accuracy',
                             percentage=False)

plt.show()

#%%
metric = "manhattan"
n_neighbors=3
weight = "distance"

knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weight)
knn.fit(trnX, trnY)
pred = knn.predict(tstX)
print("accuracy: ", metrics.accuracy_score(tstY, pred))
