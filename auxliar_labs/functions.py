#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt

plt.figure(num=1, figsize=(5,4))


# In[2]:


import pandas as pd
from pandas.plotting import register_matplotlib_converters


# In[3]:


def choose_grid(nr):
    return nr // 4 + 1, 4

def line_chart(ax: plt.Axes, series: pd.Series, title: str, xlabel: str, ylabel: str, percentage=False):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)
    ax.plot(series)


# In[4]:


def multiple_line_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, xlabel: str, ylabel: str, percentage=False):
    legend: list = []
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)

    for name, y in yvalues.items():
        ax.plot(xvalues, y)
        legend.append(name)
    ax.legend(legend, loc='best', fancybox = True, shadow = True)    


# In[5]:


def bar_chart(ax: plt.Axes, xvalues: list, yvalues: list, title: str, xlabel: str, ylabel: str, percentage=False):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(xvalues, rotation=90, fontsize='small')
    if percentage:
        ax.set_ylim(0.0, 1.0)
    ax.bar(xvalues, yvalues, edgecolor='grey')


# In[7]:


import numpy as np

def multiple_bar_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, xlabel: str, ylabel: str, percentage=False):

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    x = np.arange(len(xvalues))  # the label locations
    ax.set_xticks(x)
    ax.set_xticklabels(xvalues, fontsize='small')
    if percentage:
        ax.set_ylim(0.0, 1.0)
    width = 0.8  # the width of the bars
    step = width / len(yvalues)
    k = 0
    for name, y in yvalues.items():
        ax.bar(x + k * step, y, step, label=name)
        k += 1
    ax.legend(loc='lower center', ncol=len(yvalues), bbox_to_anchor=(0.5, -0.2), fancybox = True, shadow = True)    

    
import itertools
import matplotlib.pyplot as plt
CMAP = plt.cm.Blues

def plot_confusion_matrix(ax: plt.Axes, cnf_matrix: np.ndarray, classes_names: list, normalize: bool = False):
    if normalize:
        total = cnf_matrix.sum(axis=1)[:, np.newaxis]
        cm = cnf_matrix.astype('float') / total
        title = "Normalized confusion matrix"
    else:
        cm = cnf_matrix
        title = 'Confusion matrix'
    np.set_printoptions(precision=2)
    tick_marks = np.arange(0, len(classes_names), 1)
    ax.set_title(title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes_names)
    ax.set_yticklabels(classes_names)
    ax.imshow(cm, interpolation='nearest', cmap=CMAP)

    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt), horizontalalignment="center")
        
        
        

def get_correlations(corr_matrix, tr):
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    upper = np.absolute(upper)
    counter = 0
    conjuntos = []
    for index, row in upper.iterrows():
        if counter == 0:
            serie = row[row.ge(tr)]
            size = len(serie)
            if size > 0:
                counter = size
                correlated = serie.axes[0].tolist()
                correlated.append(row.name)
                conjuntos.append(correlated)
        else:
            counter-=1
    return conjuntos

def reduce(data, columns, name):
    e_col=[]
    for c in columns:
        if c in data.columns:
            e_col.append(c)
    
    if len(e_col) > 3:
        pdf =data[e_col]

        mean = name + "Mean"
        std = name + "Std"
        median = name + "Median"

        data[mean] = pdf.mean(axis=1)
        data[std] = pdf.std(axis=1)
        data[median] = pdf.median(axis=1)
        data = data.drop(columns=list(pdf.columns))
    
    return data

def red_correlations(df, tr=0.9, n=3):
    df = df.copy()
    corr_mtx = df.corr()
    correlations = get_correlations(corr_mtx, tr=tr)
    
    for c in correlations:
        if len(c)>n:
            df = reduce(df, c, c[0])
    
    return df


def sensitivity(tstY, prdY, labels):
    cfm = metrics.confusion_matrix(tstY, prdY, labels)
    return cfm[0,0]/(cfm[0,0]+cfm[0,1])


from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

def split_dataset(data1):
    #fazer data split para dataset com train_test_split
    y1: np.ndarray = data1['class'].values 
    X1: np.ndarray = data1.drop('class', axis=1).values
    labels1 = pd.unique(y1)
    trnX1, tstX1, trnY1, tstY1 = train_test_split(X1, y1, train_size=0.7, stratify=y1)
    
    return (trnX1, trnY1, tstX1, tstY1, labels1)


def compare_datasets(clf1, clf2, data1, data2):
    tts1 = data1
    tts2 = data2
    datasets = (tts1, tts2)
    clfs = (clf1, clf2)
    # (trnX, trnY, tstX, tstY, labels)
    
    xvalues = ["dataset1", "dataset2"]
    yvalues = []
    syvalues = []
    cnf_mtx = []
    for i,d in enumerate(datasets):
        clfs[i].fit(d[0], d[1])
        prdY = clfs[i].predict(d[2])
        yvalues.append(metrics.accuracy_score(d[3], prdY))
        cfm = metrics.confusion_matrix(d[3], prdY, d[4])
        cnf_mtx.append(cfm)
        syvalues.append(sensitivity(d[3],prdY, d[4]))
    
    plt.figure()
    plot_confusion_matrix(plt.gca(), cnf_mtx[0], datasets[0][4])
    plt.show()
    plt.figure()
    plot_confusion_matrix(plt.gca(), cnf_mtx[1], datasets[1][4])
    plt.show()
    plt.figure()
    bar_chart(plt.gca(), xvalues, yvalues, 'Comparison of Datasets', '', 'accuracy', percentage=True)
    plt.show()
    plt.figure()
    bar_chart(plt.gca(), xvalues, syvalues, 'Comparison of Datasets', '', 'sensitivity', percentage=True)
    plt.show()

from imblearn.over_sampling import SMOTE, RandomOverSampler

def smote_split(unba,  to_clf='class', train_ratio=0.7, random_state=52, in_df=False):
    data = unba.copy()
    columns = unba.columns
    y: np.ndarray = data[to_clf].values 
    X: np.ndarray = data.drop(to_clf, axis=1).values
    labels1 = pd.unique(y)
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state = random_state)
    
    RANDOM_STATE = 42
    smote = SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE)
    smote_X, smote_y = smote.fit_sample(trnX, trnY)
    smote_y = smote_y.reshape((-1, 1))
    tstY = tstY.reshape((-1, 1))
    
    if in_df:
        df = np.append(smote_X, smote_y,axis=1)
        dfa = pd.DataFrame(df, columns=columns)
        print("returning in dataframe")
        return dfa
    
    #trnX, trnY, tstX, tstY
    return (smote_X, smote_y, tstX, tstY, labels1)


from sklearn.preprocessing import Normalizer

def normalize(df, ignore):
    #ignore = ['class','gender']
    df = df.copy()
    df_nr = df[df.columns.difference(ignore)]
    df_ignored = df[ignore]
    #df_id = data_smote.loc[:,'id']
    #df_c = data_smote.loc[:,'class']
    #df_g = data_smote.loc[:,'gender']

    transf = Normalizer().fit(df_nr)
    df_nr = pd.DataFrame(transf.transform(df_nr, copy=True), columns= df_nr.columns)

    data_norm = df_nr.join(df_ignored)
    #norm_data = norm_data.join(df_g)
    #data_norm = norm_data.join(df_c, how='right')

    data_norm.head()
    return data_norm

from sklearn import preprocessing

def normalize_n(df, ignore):
    df = df.copy()
    df_nr = df[df.columns.difference(ignore)]
    df_ignored = df[ignore]
    
    x = df_nr.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns= df_nr.columns)
    
    data_norm = df.join(df_ignored)
    return data_norm


def cut(df, bins, ig_classes, cut="cut"):
    labels = list(map(str,range(1,bins+1)))
    dfc = df.copy()
    if cut == "cut":
        for col in dfc:
            if col not in ig_classes: 
                dfc[col] = pd.cut(dfc[col],bins,labels=labels)
    elif cut == "qcut":
         for col in dfc:
            if col not in ig_classes: 
                dfc[col] = pd.qcut(dfc[col],bins,labels=labels)
    return dfc
 
    
def dummy(df,ig_classes):
    dfc = df.copy()
    dummylist = []
    for att in dfc:
        if att in ig_classes: dfc[att] = dfc[att].astype('category')
        dummylist.append(pd.get_dummies(dfc[[att]]))
    dummified_df = pd.concat(dummylist, axis=1)
    return dummified_df



from mlxtend.frequent_patterns import apriori, association_rules #for ARM

def freq_itemsets(df, minpaterns=30):
    dfc = df.copy()
    frequent_itemsets = {}
    minsup = 1.0
    while minsup>0:
        minsup = minsup*0.9
        frequent_itemsets = apriori(dfc, min_support=minsup, use_colnames=True)
        if len(frequent_itemsets) >= minpaterns:
            print("\nMinimum support:",minsup)
            break
    print("Number of found patterns:",len(frequent_itemsets))
    return frequent_itemsets

def assoc_rules(fi, orderby="lift", min_confidence=0.7):
    
    rules = association_rules(fi, metric="confidence", min_threshold=min_confidence)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules = rules[(rules['antecedent_len']>=2) & (rules['confidence']>min_confidence)]
    if orderby!=None:
        rules = rules.sort_values(by = [orderby], ascending =False)
    return rules

# In[ ]:




