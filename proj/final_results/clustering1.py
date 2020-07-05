import pandas as pd
import numpy as np
from sklearn import metrics, cluster
from itertools import cycle
from proj.aux_libs import datapreprocessing as datapp, plot
import matplotlib.pyplot as plt

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

df = datapp.preprocess(data, to_clf, normalization=normalization, ignore_classes=categoric, as_df=True)
#%%
n_clusters = [2,3,4,5,6,7,8,9,10,16]

plt.figure()
fig, axs = plt.subplots(1, 2, figsize=(14, 8), squeeze=False)
inertias = []
sil = []

y: np.ndarray = df[to_clf].values
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

print(sil[3])

plot.multiple_line_chart(axs[0, 0], n_clusters, ivalues, '\nKmeans',
                         'nr estimators', 'inertia', percentage=False)
plot.multiple_line_chart(axs[0, 1], n_clusters, svalues, '\nKmeans',
                         'nr estimators', 'silhouete', percentage=False)

plt.show()


#%%
n_clusters=5
algs = ["PCA", "selectkbest"]
plt.figure()
fig, axs = plt.subplots(2 ,len(algs), figsize=(14, 8), squeeze=False)
for a in range(len(algs)):
    datar = datapp.feature_reduction(df, "class",["class","id"], n_features=2, as_int=True, alg=algs[a])

    y: np.ndarray = datar[to_clf].values
    X: np.ndarray = datar.drop([to_clf,"id"], axis=1).values

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
n_clusters=5
algs = ["PCA", "selectkbest"]
plt.figure()
fig, axs = plt.subplots(len(algs) ,1, figsize=(12, 10), squeeze=False)
for i in range(len(algs)):
    alg = algs[i]
    datar = datapp.feature_reduction(df, "class",["class","id"], n_features=2, as_int=True, alg=alg)

    y: np.ndarray = datar[to_clf].values
    X: np.ndarray = datar.drop([to_clf,"id"], axis=1).values

    kmeans_model = cluster.KMeans(n_clusters=n_clusters, random_state=rs).fit(X)
    labels = kmeans_model.labels_
    cluster_centers = kmeans_model.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    counts = {}
    c0 = []
    c1 = []
    for k in range(n_clusters_):
        my_members = labels == k
        classes = y[my_members,].astype("int")
        c = np.bincount(classes)
        c0.append(c[0])
        c1.append(c[1])
    counts[0] = c0
    counts[1] = c1

    print(counts)
    plot.multiple_bar_chart(axs[i, 0], list(range(1, n_clusters + 1)), counts, 'Cluster balance with {}'.format(alg),
                            'Clusters', 'Number of points')
plt.show()


