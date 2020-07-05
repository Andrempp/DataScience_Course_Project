import pandas as pd
from proj.aux_libs import evaluation as eval, plot
import re
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 15)

#Statistical description

rs = 32
data = pd.read_csv('../../data/pd_speech_features.csv', skiprows=[0])
to_clf = "class"

#%%
eval.general_eval(data, to_clf)
#%%
int_columns = data.select_dtypes(include='int64').columns
int_columns

#%%
#depois de análise, variáveis categóricas são
categoric = ["gender", "class"]

#%%
#describe of monthly features
col_analyze = data.loc[ : , 'mean_MFCC_0th_coef':'mean_MFCC_12th_coef'].columns
eval.general_eval(data, to_clf, col_analyze)

col_analyze2 = data.loc[ : , "mean_0th_delta":"mean_12th_delta"].columns
eval.general_eval(data, to_clf, col_analyze2)

plot.plot_distributions(data, col_analyze, bins=[20], library="seaborn")
#%%
#comparar distribuições dos vários th's
columns = data.columns
rs = [re.compile(".*_{}[th|st|nd|rd].*".format(i)) for i in range(0,13)]

conjunto = []
for r in rs:
    c = list(filter(r.match, columns))
    conjunto.append(c)

for conj in conjunto:
    plot.plot_distributions(data, conj, bins=[20], library="seaborn")


#%%
#comparar distribuições dos vários dec's

columns = data.columns
rs = [re.compile(".*dec_{}$".format(i)) for i in range(1,37)]

conjunto = []
for r in rs:
    c = list(filter(r.match, columns))
    conjunto.append(c)


for conj in conjunto:
    plot.plot_distributions(data, conj, bins=[20], library="seaborn")

#%%%
outliers_col = eval.outliers_category(data, data.columns, ratio=1.5, by="column")
outliers_row = eval.outliers_category(data, data.columns, ratio=1.5, by="row")

plt.figure()
bin = 20
plt.hist(outliers_col, bin,)
plt.title = "Frequencies of quantity of outliers (by column)"
plt.show()
plt.figure()
bin = 20
plt.hist(outliers_row, bin,cumulative=False)
plt.title = "Frequencies of quantity of outliers (by row)"
plt.show()
#plt.figure()
#plot.bar_chart(plt.gca(), data.columns, all_outliers, 'Outlier analysis', '', 'Number Outliers')
#plt.show()

#Decidir o que fazer com esta informação
#%%
