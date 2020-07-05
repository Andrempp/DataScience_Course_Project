from itertools import cycle

import numpy as np
import pandas as pd
from sklearn import metrics, cluster

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
#%%
df = datapp.subsample(data, to_clf, per_class=500)
print(df.shape)
#%%
n_clusters = [2,3,4,5,6,7,8,9,10,16]

plt.figure()
fig, axs = plt.subplots(1, 2, figsize=(14, 8), squeeze=False)
inertias = []
sil = []

y: np.ndarray = df[to_clf].values
n_classes = np.unique(y)

X: np.ndarray = df.drop(to_clf, axis=1).values

for n in n_clusters:
    kmeans = cluster.KMeans(n_clusters=n, random_state=1).fit(X)
    prdY = kmeans.labels_
    inertias.append(kmeans.inertia_)
    sil.append(metrics.silhouette_score(X, prdY))

ivalues={}
svalues={}
ivalues["norm"] = inertias
svalues["norm"] = sil
print(sil[4])

plot.multiple_line_chart(axs[0, 0], n_clusters, ivalues, '\nKmeans',
                         'nr estimators', 'inertia', percentage=False)
plot.multiple_line_chart(axs[0, 1], n_clusters, svalues, '\nKmeans',
                         'nr estimators', 'silhouete', percentage=False)

plt.show()

#%%
n_clusters=6
algs = ["PCA", "selectkbest"]
plt.figure()
fig, axs = plt.subplots(2 ,len(algs), figsize=(14, 8), squeeze=False)
for a in range(len(algs)):
    datar = datapp.feature_reduction(df, to_clf,categoric+[to_clf], n_features=2, as_int=True, alg=algs[a])

    y: np.ndarray = datar[to_clf].values
    X: np.ndarray = datar.drop([to_clf], axis=1).values

    kmeans_model = cluster.KMeans(n_clusters=n_clusters, random_state=rs).fit(X)
    labels = kmeans_model.labels_
    cluster_centers = kmeans_model.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    #plot
    #plt.clf()
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        axs[0,a].plot(X[my_members, 0], X[my_members, 1], col + '.')
        axs[0,a].plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    axs[0,a].set_title('Estimated clusters with {}'.format(algs[a]))

    for i, x in enumerate(X):
        if y[i] == 1:
            axs[1,a].scatter(x[0], x[1], c="tab:blue", label="true")
        else:
            axs[1,a].scatter(x[0], x[1], c="tab:orange", label="false")
    axs[1,a].set_title('True clusters with {}'.format(algs[a]))


plt.show()


#%%
n_clusters = 6
algs = ["PCA", "selectkbest"]
plt.figure()
fig, axs = plt.subplots(len(algs), 1, figsize=(12, 10), squeeze=False)
for i in range(len(algs)):
    alg = algs[i]
    datar = datapp.feature_reduction(df, to_clf, categoric+[to_clf], n_features=2, as_int=True, alg=alg)

    y: np.ndarray = datar[to_clf].values
    X: np.ndarray = datar.drop([to_clf], axis=1).values

    kmeans_model = cluster.KMeans(n_clusters=n_clusters, random_state=rs).fit(X)
    labels = kmeans_model.labels_
    cluster_centers = kmeans_model.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    counts_dic = {}
    for j in range(1, len(n_classes)+1):
        counts_dic[j] = []
    for k in range(n_clusters_):
        my_members = labels == k
        classes = y[my_members,].astype("int")
        unique, counts = np.unique(classes, return_counts=True)
        teste = dict(zip(unique, counts))
        print(teste)
        for a in range(1,len(n_classes)+1):
            if a in teste.keys():
                counts_dic[a].append(teste[a])
            else:
                counts_dic[a].append(0)

    print(counts_dic)
    plot.multiple_bar_chart(axs[i, 0], list(range(1, n_clusters + 1)), counts_dic, 'Cluster balance with {}'.format(alg),
                            'Clusters', 'Number of points')
plt.show()

