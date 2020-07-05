import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from proj.aux_libs import datapreprocessing as datapp, evaluation as eval, plot
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 15)
rs = 32
data = pd.read_csv('../../data/pd_speech_features.csv', skiprows=[0])


to_clf = "class"
categoric = ["gender", "id"]
to_remove = ["id"]
data.shape

normalization = "minmax"
bal = "smote"
df = datapp.preprocess(data, to_clf, normalization=normalization, ignore_classes=categoric, as_df=True)
df=data
y: np.ndarray = df[to_clf].values
X: np.ndarray = df.drop(to_clf, axis=1).values
#%%
select = SelectKBest(f_classif, k=10).fit(X, y)
ind = select.get_support(indices=True)
col = df.columns[ind].tolist()

X_new = select.transform(X)
dfk = pd.DataFrame(X_new, columns=col)
#%%
bins = list(range(3,12))
qdfs = []
cdfs = []
for b in bins:
    qdfs.append(eval.cut(dfk, b, ['class','id', 'gender'], cut="qcut"))
    cdfs.append(eval.cut(dfk, b, ['class','id', 'gender'], cut="cut"))
#%%
dummy_qdfs = []
dummy_cdfs = []
for i in range(len(bins)):
    dummy_qdfs.append(eval.dummy(qdfs[i], ['class','id','gender']))
    dummy_cdfs.append(eval.dummy(cdfs[i], ['class', 'id', 'gender']))
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
    rules_qsup.append(eval.assoc_rules(fiq_q[i], orderby="support",inverse=False, min_confidence=0.9).head(20))
    rules_csup.append(eval.assoc_rules(fiq_c[i], orderby="support", inverse=False,min_confidence=0.9).head(20))

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
