# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 23:15:37 2022

@author: @author: Ernest (Khashayar) Namdar
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as t_dist



def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color, linewidth=3)
    plt.setp(bp['whiskers'], color=color, linewidth=3)
    plt.setp(bp['caps'], color=color, linewidth=3)
    plt.setp(bp['medians'], color=color, linewidth=3)

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def ttest2(m1,m2,s1,s2,n1,n2,m0=0,equal_variance=False):
    if equal_variance is False:
        se = np.sqrt((s1**2/n1) + (s2**2/n2))
        # welch-satterthwaite df
        df = ((s1**2/n1 + s2**2/n2)**2)/((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
    else:
        # pooled standard deviation, scaled by the sample sizes
        se = np.sqrt((1/n1 + 1/n2) * ((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
        df = n1+n2-2

    t = (m1-m2-m0)/se
    dat = {"Difference of means":m1-m2, "Std Error":se, "t":t, "p-value":2*t_dist.cdf(-abs(t),df)}
    return dat


def plot_boxes(m1_mean, m1_sd, m1_5stats, m2_mean, m2_sd, m2_5stats, ticks):
    pval12 = ttest2(m1_mean, m2_mean, m1_sd, m2_sd, 30, 30)["p-value"]

    plt.figure(figsize=(10, 8))
    bpl = plt.boxplot([m1_5stats, m2_5stats], positions=np.array(range(2)) * 2.0, widths=0.6, whis=(0, 100))
    plt.scatter(np.array(range(2)) * 2.0, [m1_mean, m2_mean], c='firebrick', marker='*', s=100)

    plt.xticks(range(0, len(ticks) * 2, 2), ticks, fontsize=14)
    plt.xlim(-2, len(ticks) * 2)

    # Dynamically adjust y-axis range based on data
    all_data = m1_5stats + m2_5stats
    upper_limit = max(all_data)
    lower_limit = min(all_data)
    margin = (upper_limit - lower_limit) * 0.1  # 10% margin
    plt.ylim(lower_limit - margin, upper_limit + margin * 5)  # Adjust margin for annotation space

    # Adjust positions for statistical annotation
    positions = list(np.array(range(2)) * 2.0)
    x1, x2 = positions[0], positions[1]
    col = "midnightblue"
    y, h = upper_limit + margin * 2, margin  # dynamic height based on max value and margin
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text((x1 + x2) * .5, y + h, "p-value=" + "{:.2E}".format(pval12), ha='center', va='bottom', color=col)

    plt.ylabel("AUROC", fontsize=18)
    plt.tight_layout()
    # #plt.savefig('binWidths.svg')
