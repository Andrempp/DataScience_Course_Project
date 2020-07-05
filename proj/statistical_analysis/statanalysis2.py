import pandas as pd
from proj.aux_libs import evaluation as eval, plot
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", 15)

columns = ['Elevation','Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology'
           ,'Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
            'Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4','Soil_Type1','Soil_Type2','Soil_Type3','Soil_Type4','Soil_Type5','Soil_Type6',
            'Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type10','Soil_Type11', 'Soil_Type12', 'Soil_Type13',
            'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19','Soil_Type20',
            'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
            'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
            'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Cover_Type']

rs = 32
data = pd.read_csv('proj/data/covtype.data', names=columns)
to_clf = 'Cover_Type'
#%%
eval.general_eval(data, to_clf)
#%%
#categóricas são as 4 wilderness areas e as 40 soil types
#%%
col_analyze = data.loc[ : , 'Elevation': 'Horizontal_Distance_To_Fire_Points'].columns
eval.general_eval(data, to_clf, col_analyze)

col_analyze2 = data.loc[ : , 'Elevation': 'Horizontal_Distance_To_Fire_Points'].columns
eval.general_eval(data, to_clf, col_analyze2)

plot.plot_distributions(data, col_analyze, bins=[20], library="seaborn")
#%%
heat_data = data
fig = plt.figure(figsize=[12, 12])
corr_mtx = heat_data.corr()
sns.heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=False, cmap='Blues')
plt.title('Correlation analysis')
plt.show()

#Não está nada correlacionado, WTF, logo não fazemos redução de variaveis. mas com 581012 rows temos que subsample.
#%%%
#Análise dos outliers
outliers_col = eval.outliers_category(data, data.columns, ratio=1.5, by="column")
outliers_row = eval.outliers_category(data, data.columns, ratio=1.5, by="row")

plt.figure()
bin = 20
plt.hist(outliers_col, bin,)
plt.title = "Frequencies of quantity of outliers (by column)"
plt.show()
plt.figure()
bin = 20
plt.hist(outliers_row, bin,cumulative=False,label="Number of outliers")
plt.ylabel("Number of outliers")
plt.xlabel("Variables")
plt.show()