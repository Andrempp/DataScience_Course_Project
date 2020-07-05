#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import scipy.stats as _stats
import sklearn.metrics as metrics


plt.figure(num=1, figsize=(5, 4))

# In[2]:


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


def multiple_line_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, xlabel: str, ylabel: str,
                        percentage=False):
    legend: list = []
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)

    for name, y in yvalues.items():
        ax.plot(xvalues, y)
        legend.append(name)
    ax.legend(legend, loc='best', fancybox=True, shadow=True)


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




def multiple_bar_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, xlabel: str, ylabel: str,
                       percentage=False):
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
    ax.legend(loc='lower center', ncol=len(yvalues), bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True)




CMAP = plt.cm.Blues


def plot_confusion_matrix(ax: plt.Axes, cnf_matrix: np.ndarray, classes_names: list, normalize: bool = False,
                          title_m=None):
    if normalize:
        total = cnf_matrix.sum(axis=1)[:, np.newaxis]
        cm = cnf_matrix.astype('float') / total
        title = "Normalized confusion matrix"
    else:
        cm = cnf_matrix
        if title_m == None:
            title = 'Confusion matrix'
        else:
            title = title_m
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



def compute_known_distributions(x_values, n_bins) -> dict:
    distributions = dict()
    # Gaussian
    mean, sigma = _stats.norm.fit(x_values)
    distributions['Normal(%.1f,%.2f)'%(mean,sigma)] = _stats.norm.pdf(x_values, mean, sigma)
    # LogNorm
    #  sigma, loc, scale = _stats.lognorm.fit(x_values)
    #  distributions['LogNor(%.1f,%.2f)'%(np.log(scale),sigma)] = _stats.lognorm.pdf(x_values, sigma, loc, scale)
    # Exponential
    loc, scale = _stats.expon.fit(x_values)
    distributions['Exp(%.2f)'%(1/scale)] = _stats.expon.pdf(x_values, loc, scale)
    # SkewNorm
    # a, loc, scale = _stats.skewnorm.fit(x_values)
    # distributions['SkewNorm(%.2f)'%a] = _stats.skewnorm.pdf(x_values, a, loc, scale)
    return distributions

def histogram_with_distributions(ax: plt.Axes, series: pd.Series, var: str):
    values = series.sort_values().values
    n, bins, patches = ax.hist(values, 20, density=True, edgecolor='grey')
    distributions = compute_known_distributions(values, bins)
    multiple_line_chart(ax, values, distributions, 'Best fit for %s'%var, var, 'probability')


def plot_distributions(df, columns, library="matplot", bins=range(5, 100, 20)):
    data = df.copy()
    if library=="seaborn":
        rows, cols = choose_grid(len(columns)-1)
        plt.figure()
        fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)
        i, j = 0, 0
        for n in range(len(columns)):
            histogram_with_distributions(axs[i, j], data[columns[n]].dropna(), columns[n])
            i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
        fig.tight_layout()
        plt.show()
    elif library=="matplot":
        rows = len(columns)
        cols = 5
        plt.figure()
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
        for i in range(len(columns)):
            for j in range(len(bins)):
                axs[i, j].set_title('Histogram for %s' % columns[i])
                axs[i, j].set_xlabel(columns[i])
                axs[i, j].set_ylabel("probability")
                axs[i, j].hist(data[columns[i]].dropna().values, bins[j])
        fig.tight_layout()
        plt.show()
    else:
        raise ValueError("Argument library must be 'seaborn' or 'matplot'")



def compare_datasets(clfs, datasets):
    #tts1 = data1
    #tts2 = data2
    #datasets = (tts1, tts2)
    #clfs = (clf1, clf2)
    # (trnX, trnY, tstX, tstY, labels)

    xvalues = list(map(str,list(range(0,len(datasets)))))
    yvalues = []
    cnf_mtx = []
    for i, d in enumerate(datasets):
        clfs[i].fit(d[0], d[1])
        prdY = clfs[i].predict(d[2])
        yvalues.append(metrics.accuracy_score(d[3], prdY))
        cfm = metrics.confusion_matrix(d[3], prdY, d[4])
        cnf_mtx.append(cfm)

    for i in range(len(datasets)):
        plt.figure()
        plot_confusion_matrix(plt.gca(), cnf_mtx[i], datasets[i][4])
        plt.show()
    print("mekielas em chelas")
    plt.figure()
    bar_chart(plt.gca(), xvalues, yvalues, 'Comparison of Datasets', '', 'accuracy', percentage=True)
    plt.show()



