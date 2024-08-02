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
from itertools import product
import copy
import warnings
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import re


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


def val_experiment_gridsearch_kfold(model_name, X_dev, y_dev, k, hypers=None):
    hyper_params = copy.deepcopy(hypers)
    experiments_performances = []
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    i = 0
    for train_index, test_index in skf.split(X_dev, y_dev):
        # print("Woring on fold#", fld)
        X_train, X_val = X_dev.iloc[train_index], X_dev.iloc[test_index]
        y_train, y_val = y_dev.iloc[train_index], y_dev.iloc[test_index]
        if hyper_params is None:
            exec("clf = eval(model_name)(random_state=i)")
        else:
            hyper_params["random_state"] = i
            #exec("clf = eval(model_name)(**hyper_params)")
            clf = eval(model_name)(**hyper_params)
        clf.fit(X_train, y_train)
        predictions = clf.predict_proba(X_val)[:,1]
        experiments_performances.append(roc_auc_score(y_val, predictions))
        i += 1
        #print(roc_auc_score(y_val, predictions))

    mean_perf = np.mean(experiments_performances)
    return mean_perf


def grid_search_kfold(model_name, param_grid, X_dev, y_dev, k):
    params = param_grid.keys()
    best_params = {}
    for param in params:
        best_params[param] = None
    best_performance = 0
    combinations = [dict(zip(param_grid, v)) for v in product(*param_grid.values())]
    for comb in combinations:
        performance = val_experiment_gridsearch_kfold(model_name, X_dev, y_dev, k, hypers=comb)
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
    k = 10  # Number of folds
    data_sheet_path = "../data/kyc.csv" # know your customer sheet that contains the GT labels
    data_sheet_df = pd.read_csv(data_sheet_path)

    ##############################################################
    # One-Hot Encoding for Gender
    Gender_dummies = pd.get_dummies(data_sheet_df['Gender'], prefix='Gender')
    data_sheet_df = pd.concat([data_sheet_df, Gender_dummies], axis=1)
    data_sheet_df.drop('Gender', axis=1, inplace=True)

    # One-Hot Encoding for Occupation
    occupation_dummies = pd.get_dummies(data_sheet_df['Occupation'], prefix='Occupation')
    data_sheet_df = pd.concat([data_sheet_df, occupation_dummies], axis=1)
    data_sheet_df.drop('Occupation', axis=1, inplace=True)
    data_sheet_df = data_sheet_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    #########################Experiment1####################################
    balanced_risk_sheet_df = balance_dset(data_sheet_df, "label") # make the dataset balanced

    y = balanced_risk_sheet_df['label']
    to_drop = ["Name", "cust_id", "label"]
    X = balanced_risk_sheet_df.drop(to_drop, axis=1)

    lgbm_clf_param_grid = {
        "n_estimators": [50, 100, 200],  # Number of boosted trees to fit
        "learning_rate": [0.01, 0.1, 0.2],  # Boosting learning rate
        #"num_leaves": [31, 62, 127],  # Maximum tree leaves for base learners
        "max_depth": [5, -1]  # Maximum tree depth for base learners, -1 means no limit
        # LightGBM does not use 'depth' parameter as in CatBoost. Instead, 'max_depth' is used for control over tree depth.
        # "reg_lambda": [0, 1, 10],  # L2 regularization term on weights, equivalent to 'l2_leaf_reg' in CatBoost
        # "max_bin": [255, 510],  # Number of bins that feature values will be bucketed in. Small number of bins may reduce training accuracy but can deal with overfitting.
        # "boosting_type": ['gbdt', 'dart'],  # 'gbdt' - traditional Gradient Boosting Decision Tree, 'dart' - Dropouts meet Multiple Additive Regression Trees
    }

    start = time.time()
    print("running LightGBM")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)

    experiment1_aucs = []
    fld = 0
    # Split the dataset into k consecutive folds (without shuffling by default).
    for train_index, test_index in skf.split(X, y):
        print("Woring on fold#", fld)
        X_dev, X_test = X.iloc[train_index], X.iloc[test_index]
        y_dev, y_test = y.iloc[train_index], y.iloc[test_index]
        best_params = grid_search_kfold("LGBMClassifier", lgbm_clf_param_grid, X_dev, y_dev, k-1)
        best_params["random_state"] = fld
        auc = test_experiment("LGBMClassifier", X_dev, y_dev, X_test, y_test, best_params)
        experiment1_aucs.append(auc)
        fld += 1
    min_experiment1, q1_experiment1, med_experiment1, q3_experiment1, max_experiment1 = five_number_summary(experiment1_aucs)
    mean_experiment1 = np.mean(experiment1_aucs)
    sd_experiment1 = np.std(experiment1_aucs)
    print("experiment1 stats (mean, sd, 5stats):", [mean_experiment1, sd_experiment1, min_experiment1, q1_experiment1, med_experiment1, q3_experiment1, max_experiment1])
    end = time.time()
    duration = end-start
    print("The run was completed in: ", int(duration/60), "minutes and ", int(duration%60), "seconds")
    input()
    #########################Experiment2####################################
    tabnet_param_grid = {
        'max_epochs': [100],  # Could add more values
        'optimizer_params': [{'lr': 5e-2}],  # Learning rate
        'batch_size': [256],  # Could vary this
        'virtual_batch_size': [128]  # Size of the mini batches for ghost batch normalization
    }

    start = time.time()
    print("Running TabNet")
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    
    experiment2_aucs = []
    fld = 0

    # simplified process, assuming a single set of parameters
    for train_index, test_index in skf.split(X, y):
        print("Working on fold#", fld)
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
        
        # Instantiate model
        clf = TabNetClassifier(optimizer_params=tabnet_param_grid['optimizer_params'][0])
        
        # Fit model
        clf.fit(X_train.values, y_train,
                eval_set=[(X_train.values, y_train), (X_valid.values, y_valid)],
                eval_name=['train', 'valid'],
                eval_metric=['auc'],
                batch_size=tabnet_param_grid['batch_size'][0],
                virtual_batch_size=tabnet_param_grid['virtual_batch_size'][0],
                max_epochs=tabnet_param_grid['max_epochs'][0],
                patience=40)

        # Predict and evaluate
        preds_proba = clf.predict_proba(X_valid.values)[:, 1]
        auc = roc_auc_score(y_valid, preds_proba)
        experiment2_aucs.append(auc)
        fld += 1

    # Calculate statistics
    min_experiment2, q1_experiment2, med_experiment2, q3_experiment2, max_experiment2 = np.percentile(experiment2_aucs, [0, 25, 50, 75, 100])
    mean_experiment2 = np.mean(experiment2_aucs)
    sd_experiment2 = np.std(experiment2_aucs)
    print("experiment2 stats (mean, sd, 5stats):", [mean_experiment2, sd_experiment2, min_experiment2, q1_experiment2, med_experiment2, q3_experiment2, max_experiment2])

    end = time.time()
    duration = end - start
    print("The run was completed in: ", int(duration // 60), "minutes and ", int(duration % 60), "seconds")
    

    plot_boxes(mean_experiment1, sd_experiment1, [min_experiment1, q1_experiment1, med_experiment1, q3_experiment1, max_experiment1],
                mean_experiment2, sd_experiment2, [min_experiment2, q1_experiment2, med_experiment2, q3_experiment2, max_experiment2],
                ["LightGBM", "TabNet"])