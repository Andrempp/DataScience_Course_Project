import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

from proj.aux_libs import datapreprocessing as datapp, plot
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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
# %%
# PARAMETERS PARA TESTAR: criterion, splitter, max_depth, min_sample_leaf, min_samples_split

min_samples_leaf = [.05, .025, .01, .0075, .005, .0025, .001]
max_depths = [5, 10, 25, 50]
criteria = ['entropy', 'gini']

plt.figure()
fig, axs = plt.subplots(1, len(criteria), figsize=(12, 7), squeeze=False)
for k in range(len(criteria)):
    f = criteria[k]
    values = {}
    for d in max_depths:
        yvalues = []
        for n in min_samples_leaf:
            tree = DecisionTreeClassifier(min_samples_leaf=n, max_depth=d, criterion=f, random_state=rs)
            tree.fit(trnX, trnY)
            pred = tree.predict(valX)
            yvalues.append(metrics.accuracy_score(valY, pred))
        values[d] = yvalues
    plot.multiple_line_chart(axs[0, k], min_samples_leaf, values, 'Decision Trees with %s criteria' % f,
                             'min_samples_leaf', 'accuracy', percentage=True)

plt.show()

# %%
criterion = "gini"
# %%


min_samples_leaf = [.01, .0075, .005, .0025, .001]
max_depths = [5, 10, 25]
splitters = ['best', 'random']

plt.figure()
fig, axs = plt.subplots(1, len(splitters), figsize=(12, 7), squeeze=False)
for k in range(len(splitters)):
    f = splitters[k]
    values = {}
    for d in max_depths:
        yvalues = []
        for n in min_samples_leaf:
            tree = DecisionTreeClassifier(min_samples_leaf=n, max_depth=d, criterion=criterion, splitter=f,
                                          random_state=rs)
            tree.fit(trnX, trnY)
            pred = tree.predict(valX)
            yvalues.append(metrics.accuracy_score(valY, pred))
        values[d] = yvalues
    plot.multiple_line_chart(axs[0, k], min_samples_leaf, values, 'Decision Trees with %s splitter' % f,
                             'min_samples_leaf', 'accuracy')

plt.show()
# %%
splitter = "best"

# %%

min_samples_leaf = [.01, .0075, .005, .0025, .001]
max_depths = [5, 10, 25]
min_samples_split = [2, 3, 4]

plt.figure()
fig, axs = plt.subplots(1, len(min_samples_split), figsize=(12, 5), squeeze=False)
for k in range(len(min_samples_split)):
    f = min_samples_split[k]
    values = {}
    for d in max_depths:
        yvalues = []
        for n in min_samples_leaf:
            tree = DecisionTreeClassifier(min_samples_leaf=n, max_depth=d, criterion=criterion, splitter=splitter,
                                          random_state=rs, min_samples_split=f)
            tree.fit(trnX, trnY)
            pred = tree.predict(valX)
            yvalues.append(metrics.accuracy_score(valY, pred))
        values[d] = yvalues
    plot.multiple_line_chart(axs[0, k], min_samples_leaf, values, 'Decision Trees with %s splitter' % f,
                             'min_samples_leaf', 'accuracy')

plt.show()

# INDIFERENTE, DEIXAMOS DEFUALT 2
# %%
splitter = "best"
criterion = "entropy"
max_depth = 25
min_samples_leaf = 0.001

tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, max_depth=max_depth, criterion=criterion,
                              splitter=splitter, random_state=rs)
tree.fit(trnX, trnY)
pred = tree.predict(tstX)
print("accuracy: ", metrics.accuracy_score(tstY, pred))
