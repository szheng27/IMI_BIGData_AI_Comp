#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Jan 2024

Team 37
2023-24 IMI BIGDataAIHUB Case Competition

@author: Ernest (Khashayar) Namdar
"""

# Importing the required libraries ############################################

import numpy as np
import pickle
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # load the saved AUCs
    lightgbm_aucs = pickle.load(open("../data/lightgbm_aucs.p", "rb"))
    n_training_examples = pickle.load(open("../data/n_training_examples.p", "rb"))
    n_thresholds = len(lightgbm_aucs)

    meanAUC_lightgbm = []
    minAUC_lightgbm = []
    maxAUC_lightgbm = []
    len_training = []
    for i in range(n_thresholds):
        meanAUC_lightgbm.append(np.mean(lightgbm_aucs[i]))
        minAUC_lightgbm.append(np.min(lightgbm_aucs[i]))
        maxAUC_lightgbm.append(np.max(lightgbm_aucs[i]))
        len_training.append(np.mean(n_training_examples[i]))

    fig, ax = plt.subplots(1)
    ax.set_title("Test AUC vs dataset size")
    ax.plot(len_training, meanAUC_lightgbm, linestyle='-', lw=2, color='b', label='Test mean AUC (LightGBM)', alpha=.8)
    ax.fill_between(len_training, minAUC_lightgbm, maxAUC_lightgbm, color='b', alpha=0.1)


    ax.set(xlim=[0, np.max(len_training)], ylim=[0.5, 1.0])
    ax.legend(loc="lower right")
    ax.set(ylabel='AUC')
    ax.legend(loc="lower right")
    ax.set(xlabel ='Training Dataset Size', ylabel='AUC')

    print("mean AUC for lightgbm:", np.mean(meanAUC_lightgbm), "with average range of", np.mean(np.array(maxAUC_lightgbm)-np.array(minAUC_lightgbm)))
