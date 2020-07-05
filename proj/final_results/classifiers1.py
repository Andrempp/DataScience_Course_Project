import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from proj.aux_libs import datapreprocessing as datapp, evaluation as eval, plot
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 15)
rs = 32
data = pd.read_csv('../../data/pd_speech_features.csv', skiprows=[0])
to_clf = "class"
categoric = ["gender", "id"]
to_remove = ["id"]

normalization = "standard"
bal = "smote"

accs = []
std_accs = []
sensis = []
std_sensis = []
cfms = []

labels = pd.unique(data[to_clf].values)
#%%
#NAIVE BAYES
tr=0.9
f= "PCA"
selectk = 1
datared = datapp.preprocess_alt(data, "class", red_corr=True, tr=tr, n=5, normalization=normalization,
                              ignore_classes=categoric, as_df=True)
df = datapp.feature_reduction(datared, "class",["class","id"], n_features=selectk, alg=f)
nb = GaussianNB()

acc, sens, cfm,std_acc, std_sens = eval.train_predict_kfold(df, "class", nb, bal=bal, std=True)

accs.append(acc)
std_accs.append(std_acc)
sensis.append(sens)
std_sensis.append(std_sens)
cfms.append(cfm)

print("Naive Bayes")
print("Accuracy: {} with Standard Deviation: {}\nSensitivity: {} with Standard Deviation: {}\n"
      .format(acc,std_acc,sens, std_sens))

#%%
#Decision Tree
tr=0.95
f= "selectkbest"
selectk = 1
criterion = "entropy"
splitter = "best"
max_depth = 25
min_samples_leaf = .005

datared = datapp.preprocess_alt(data, "class", red_corr=True, tr=tr, n=5, normalization=normalization,
                              ignore_classes=categoric, as_df=True)
df = datapp.feature_reduction(datared, "class",["class","id"], n_features=selectk, alg=f)

tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, max_depth=max_depth, criterion=criterion,
                              random_state=rs, splitter=splitter)


acc, sens, cfm, std_acc, std_sens = eval.train_predict_kfold(df, "class", tree, bal=bal, std=True)

accs.append(acc)
std_accs.append(std_acc)
sensis.append(sens)
std_sensis.append(std_sens)
cfms.append(cfm)

print("Decision Tree")
print("Accuracy: {} with Standard Deviation: {}\nSensitivity: {} with Standard Deviation: {}\n"
      .format(acc,std_acc,sens, std_sens))

#%%
#RandomForest
max_features="log2"
min_samples_split=4
n_estimators=50
criterion="entropy"
max_depth = 25

tr=0.95
f= "selectkbest"
selectk = 0.6

datared = datapp.preprocess_alt(data, "class", red_corr=True, tr=tr, n=5, normalization=normalization,
                              ignore_classes=categoric, as_df=True)
df = datapp.feature_reduction(datared, "class",["class","id"], n_features=selectk, alg=f)

rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
                                    criterion=criterion, random_state=rs, min_samples_split= min_samples_split)

acc, sens, cfm, std_acc, std_sens = eval.train_predict_kfold(df, "class", rf, bal=bal, std=True)

accs.append(acc)
std_accs.append(std_acc)
sensis.append(sens)
std_sensis.append(std_sens)
cfms.append(cfm)

print("Random Forest")
print("Accuracy: {} with Standard Deviation: {}\nSensitivity: {} with Standard Deviation: {}\n"
      .format(acc,std_acc,sens, std_sens))

#%%
#KNN
metric = "manhattan"
n_neighbors=1

tr=0.9
f= "selectkbest"
selectk = 1

datared = datapp.preprocess_alt(data, "class", red_corr=True, tr=tr, n=5, normalization=normalization,
                              ignore_classes=categoric, as_df=True)
df = datapp.feature_reduction(datared, "class",["class","id"], n_features=selectk, alg=f)

knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
acc, sens, cfm, std_acc, std_sens = eval.train_predict_kfold(df, "class", knn, bal=bal, std=True)

accs.append(acc)
std_accs.append(std_acc)
sensis.append(sens)
std_sensis.append(std_sens)
cfms.append(cfm)

print("KNN")
print("Accuracy: {} with Standard Deviation: {}\nSensitivity: {} with Standard Deviation: {}\n"
      .format(acc,std_acc,sens, std_sens))
#%%

tr=0.9
f= "selectkbest"
selectk = 0.9
loss = "exponential"
lr = 0.3
min_samples_leaf = 0.01
n_estimators = 100
max_depth = 3

datared = datapp.preprocess_alt(data, "class", red_corr=True, tr=tr, n=5, normalization=normalization,
                              ignore_classes=categoric, as_df=True)
df = datapp.feature_reduction(datared, "class",["class","id"], n_features=selectk, alg=f)


gb = GradientBoostingClassifier(min_samples_leaf=min_samples_leaf, n_estimators=n_estimators, loss=loss,
                                learning_rate=lr,random_state=rs, max_depth=max_depth)


acc, sens, cfm ,std_acc, std_sens= eval.train_predict_kfold(df, "class", gb, bal=bal, std=True)

accs.append(acc)
std_accs.append(std_acc)
sensis.append(sens)
std_sensis.append(std_sens)
cfms.append(cfm)

print("Gradient Boosting")
print("Accuracy: {} with Standard Deviation: {}\nSensitivity: {} with Standard Deviation: {}"
      .format(acc,std_acc,sens, std_sens))


#%%
names = ["NB", "DecisionTree", "RandomForest", "KNN", "GradientBoosting"]
evals = [accs, sensis]
stds = [std_accs, std_sensis]


plt.figure()
fig, axs = plt.subplots(1,2, figsize=(14, 6), squeeze=False)
for k in range(len(evals)):
    ev = evals[k]
    ax = axs[0,k]
    ax.errorbar(range(len(names)), ev, yerr=stds[k], fmt='-o')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    if evals[k] == accs:
        ax.set_title('Classifiers accuracy')
    if evals[k] == sensis:
        ax.set_title('Classifiers sensitivity')
    if k == 0:
        ax.set_ylabel("Accuracy")
    else:
        ax.set_ylabel("Sensitivity")
    ax.set_xlabel("Classifiers")
    ax.set_ylim(0, 1.0)
    #plt.locator_params(nbins=4)
plt.show()

plt.figure()
fig, axs = plt.subplots(1,len(cfms), figsize=(14, 6), squeeze=False)
for m in range(len(cfms)):
    plot.plot_confusion_matrix(axs[0, m], cfms[m].astype("int"), labels, title_m=names[m])
plt.show()


