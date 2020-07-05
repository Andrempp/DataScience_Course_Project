import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

from proj.aux_libs import datapreprocessing as datapp, evaluation as eval, plot
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
tstY = tstY.astype("int")

X = np.concatenate((trnX, tstX, valX))
y = np.concatenate((trnY, tstY, valY))
#%%
select = SelectKBest(f_classif, k=10).fit(X, y)
ind = select.get_support(indices=True)
col = data.columns[ind].tolist()

X_new = select.transform(X)
dfk = pd.DataFrame(X_new, columns=col)
print(dfk.columns)
#%%
dfk.head()
#%%
bins = list(range(3,12))
qdfs = []
cdfs = []
for b in bins:
    qdfs.append(eval.cut(dfk, b, categoric+[to_clf], cut="qcut"))
    cdfs.append(eval.cut(dfk, b, categoric+[to_clf], cut="cut"))
#%%
cat = ['Wilderness_Area1', 'Wilderness_Area4', 'Soil_Type3','Soil_Type10', 'Soil_Type38', 'Soil_Type39']
#%%
"""
al_dum_q = []
al_dum_c = []
for i in range(len(bins)):
    al_dum_q.append(qdfs[i][cat])
    qdfs[i] = qdfs[i].drop([cat], axis=1)
"""
#%%
dummy_qdfs = []
dummy_cdfs = []
for i in range(len(bins)):
    dummy_qdfs.append(eval.dummy(qdfs[i], categoric+[to_clf]))
    dummy_cdfs.append(eval.dummy(cdfs[i], categoric+[to_clf]))
#%%
dummy_qdfs[0].head()
#%%
fiq_q = []
fiq_c =[]
for i in range(len(bins)):
    fiq_q.append(eval.freq_itemsets(dummy_qdfs[i], minpaterns=100))
    fiq_c.append(eval.freq_itemsets(dummy_cdfs[i], minpaterns=100))
#%%
rules_q = []
rules_c = []
for i in range(len(bins)):
    rules_q.append(eval.assoc_rules(fiq_q[i], orderby="lift", min_confidence=0.9).head(20))
    rules_c.append(eval.assoc_rules(fiq_c[i], orderby="lift", min_confidence=0.9).head(20))

rules_qsup = []
rules_csup = []
for i in range(len(bins)):
    rules_qsup.append(eval.assoc_rules(fiq_q[i], orderby="support", min_confidence=0.9).head(20))
    rules_csup.append(eval.assoc_rules(fiq_c[i], orderby="support", min_confidence=0.9).head(20))

#%%
q_lifts = []
c_lifts = []
for i in range(len(bins)):
    q_lifts.append(rules_q[i]["lift"].mean())
    c_lifts.append(rules_c[i]["lift"].mean())

q_sup = []
c_sup = []
for i in range(len(bins)):
    q_sup.append(rules_qsup[i]["support"].mean())
    c_sup.append(rules_csup[i]["support"].mean())
#%%
lvalues={}
lvalues["cut"] = c_lifts
lvalues["qcut"] = q_lifts

svalues = {}
svalues["cut"] = c_sup
svalues["qcut"] = q_sup

plt.figure()
fig, axs = plt.subplots(1, 2, figsize=(12, 6), squeeze=False)
axs[0,0].set_xticks(bins)
axs[0,1].set_xticks(bins)
plot.multiple_line_chart(axs[0, 0], bins, lvalues, 'Lift of top rules of corresponding bins', 'bins', 'lift')
plot.multiple_line_chart(axs[0, 1], bins, svalues, 'Support of top rules of corresponding bins', 'bins', 'support')

plt.show()
