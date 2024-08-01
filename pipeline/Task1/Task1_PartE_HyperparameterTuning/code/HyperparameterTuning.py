#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V1 Created in Dec 2023

Team 37
2023-2024 IMI Big Data and Artificial Intelligence Competition
@author: Ernest (Khashayar) Namdar
"""

# Importing the required libraries ############################################
"""
Note: We could accelerate the operations using cuDF and cuML from RAPIDS
Nonetheless, team members had installation issues depending on the platform they used
"""

import pandas as pd #replace with cudf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time
from numpy import percentile
from boxplots import plot_boxes
from sklearn.ensemble import RandomForestClassifier
from itertools import product
import copy
import warnings
warnings.filterwarnings("ignore")


def balance_dset(df, target):
    high_risk_data = df[df[target]==1].copy()
    labels = df[target].copy()
    all_low_risk = labels[labels == 0]
    low_risk_to_keep = np.random.choice(all_low_risk.index, size=high_risk_data.shape[0], replace=False)
    low_risk_data = df.iloc[low_risk_to_keep].copy()
    new_df = pd.concat([high_risk_data, low_risk_data], axis=0)
    return new_df


def supervised_model_training(X, y, clf, seed=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed) #randomly spliting the data to train/test
    clf.fit(X_train, y_train)
    predictions_lbl = clf.predict(X_test)
    predictions_pr = clf.predict_proba(X_test)[:, 1]
    return clf, roc_auc_score(y_test, predictions_pr)


def five_number_summary(lst):
    quartiles = percentile(lst, [25,50,75])
    data_min, data_max = min(lst), max(lst)
    return data_min, quartiles[0], quartiles[1], quartiles[2], data_max


def val_experiment_gridsearch(model_name, X_dev, y_dev, N, hypers=None, test_size=0.25):
    hyper_params = copy.deepcopy(hypers)
    experiments_performances = []
    for i in range(N):
        X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=test_size, random_state=i)
        if hyper_params is None:
            exec("clf = eval(model_name)(random_state=i)")
        else:
            hyper_params["random_state"] = i
            #exec("clf = eval(model_name)(**hyper_params)")
            clf = eval(model_name)(**hyper_params)
        clf.fit(X_train, y_train)
        predictions = clf.predict_proba(X_val)[:,1]
        experiments_performances.append(roc_auc_score(y_val, predictions))
        #print(roc_auc_score(y_val, predictions))

    mean_perf = np.mean(experiments_performances)
    return mean_perf


def grid_search(model_name, param_grid, X_dev, y_dev, N):
    params = param_grid.keys()
    best_params = {}
    for param in params:
        best_params[param] = None
    best_performance = 0
    combinations = [dict(zip(param_grid, v)) for v in product(*param_grid.values())]
    for comb in combinations:
        performance = val_experiment_gridsearch(model_name, X_dev, y_dev, N, hypers=comb, test_size=0.25)
        if performance > best_performance:
            best_performance = performance
            best_params = comb
    return best_params


def test_experiment(model_name, X_dev, y_dev, X_test, y_test, hypers):
    clf = eval(model_name)(**hypers)
    clf.fit(X_dev, y_dev)
    predictions = clf.predict_proba(X_test)[:,1]
    test_perf = roc_auc_score(y_test, predictions)
    return test_perf


if __name__ == "__main__":
    N = 30 #Number of experiments
    N_val = 30 #Number of val experiments for hyprparameter tuning
    data_sheet_path = "../data/kyc.csv" # know your customer sheet that contains the GT labels
    data_sheet_df = pd.read_csv(data_sheet_path)

    ##############################################################
    # Encoding
    # Label Encoding for Gender
    data_sheet_df['Gender'] = data_sheet_df['Gender'].astype('category').cat.codes

    # Label Encoding for Occupation
    data_sheet_df['Occupation'] = data_sheet_df['Occupation'].astype('category').cat.codes

    #########################Experiment1####################################
    balanced_risk_sheet_df = balance_dset(data_sheet_df, "label") # make the dataset balanced

    y = balanced_risk_sheet_df['label']
    to_drop = ["Name", "cust_id", "label"]
    X = balanced_risk_sheet_df.drop(to_drop, axis=1)

    start = time.time()
    print("running RF with default hyperparams on balanced Dataset")
    experiment1_aucs = []
    for i in range(N):
        clf = RandomForestClassifier(random_state=i)
        clf, auc = supervised_model_training(X, y, clf, seed=i)
        experiment1_aucs.append(auc)
    min_experiment1, q1_experiment1, med_experiment1, q3_experiment1, max_experiment1 = five_number_summary(experiment1_aucs)
    mean_experiment1 = np.mean(experiment1_aucs)
    sd_experiment1 = np.std(experiment1_aucs)
    print("experiment1 stats (mean, sd, 5stats):", [mean_experiment1, sd_experiment1, min_experiment1, q1_experiment1, med_experiment1, q3_experiment1, max_experiment1])
    end = time.time()
    duration = end-start
    print("The run was completed in: ", int(duration/60), "minutes and ", int(duration%60), "seconds")

    #########################Experiment2####################################
    rf_clf_param_grid = {
        "n_estimators":[50, 100, 200],
        "max_features":['auto', 'sqrt'],
        "max_depth": [None, 5, 10]}
    start = time.time()
    print("running RF with hyperparam tuning on balanced Dataset")
    experiment2_aucs = []
    for i in range(N):
        print("Woring on experiment#", i)
        X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.25, random_state=i)
        best_params = grid_search("RandomForestClassifier", rf_clf_param_grid, X_dev, y_dev, N_val)
        best_params["random_state"] = i
        auc = test_experiment("RandomForestClassifier", X_dev, y_dev, X_test, y_test, best_params)
        experiment2_aucs.append(auc)
    min_experiment2, q1_experiment2, med_experiment2, q3_experiment2, max_experiment2 = five_number_summary(experiment2_aucs)
    mean_experiment2 = np.mean(experiment2_aucs)
    sd_experiment2 = np.std(experiment2_aucs)
    print("experiment2 stats (mean, sd, 5stats):", [mean_experiment2, sd_experiment2, min_experiment2, q1_experiment2, med_experiment2, q3_experiment2, max_experiment2])
    end = time.time()
    duration = end-start
    print("The run was completed in: ", int(duration/60), "minutes and ", int(duration%60), "seconds")
    plot_boxes(mean_experiment1, sd_experiment1, [min_experiment1, q1_experiment1, med_experiment1, q3_experiment1, max_experiment1],
                mean_experiment2, sd_experiment2, [min_experiment2, q1_experiment2, med_experiment2, q3_experiment2, max_experiment2],
                ["Default RF", "Tuned RF"])