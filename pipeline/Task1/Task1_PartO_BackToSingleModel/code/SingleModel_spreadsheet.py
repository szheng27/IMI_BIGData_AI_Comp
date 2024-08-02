#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V1 Created in Jan 2024

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
from itertools import product
import copy
import warnings
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import re
from sklearn.utils import resample
import sqlite3
from sklearn.metrics import confusion_matrix
from cf_matrix import make_confusion_matrix
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb

warnings.filterwarnings("ignore")


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
    predictions = clf.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    make_confusion_matrix(cm, group_names=['0', '1'])
    plt.show()
    predictions = clf.predict_proba(X_test)[:, 1]
    print("AUC score:", roc_auc_score(y_test, predictions))
    return clf


def revised_undersample_data(X, y, seed=0):
    # Combine the feature matrix with the target vector
    data = pd.concat([X, y], axis=1)
    
    # Separate the majority and minority classes
    majority_class = data[data[y.name] == data[y.name].mode()[0]]
    minority_class = data[data[y.name] != data[y.name].mode()[0]]
    
    # Downsample the majority class
    majority_downsampled = resample(majority_class,
                                    replace=False,  # sample without replacement
                                    n_samples=len(minority_class),  # match minority class size
                                    random_state=seed)  # reproducible results
    
    # Combine minority class with downsampled majority class
    downsampled_data = pd.concat([majority_downsampled, minority_class])
    
    # Find the unused part of the majority class
    unused_majority = majority_class.drop(majority_downsampled.index)
    
    # Splitting back into X and y for downsampled data
    X_downsampled = downsampled_data.drop(y.name, axis=1)
    y_downsampled = downsampled_data[y.name]
    
    # Splitting back into X and y for unused majority data
    X_unused_majority = unused_majority.drop(y.name, axis=1)
    y_unused_majority = unused_majority[y.name]
    
    return X_downsampled, y_downsampled, X_unused_majority, y_unused_majority



if __name__ == "__main__":
    k = 10  # Number of folds

    lgbm_clf_param_grid = {
        "n_estimators": [50, 100, 200],  # Number of boosted trees to fit
        "learning_rate": [0.01, 0.1, 0.2],  # Boosting learning rate
        "num_leaves": [31, 62, 127],  # Maximum tree leaves for base learners
        "max_depth": [5, -1],  # Maximum tree depth for base learners, -1 means no limit
        # LightGBM does not use 'depth' parameter as in CatBoost. Instead, 'max_depth' is used for control over tree depth.
        "reg_lambda": [0, 1, 10],  # L2 regularization term on weights, equivalent to 'l2_leaf_reg' in CatBoost
        "max_bin": [255, 510],  # Number of bins that feature values will be bucketed in. Small number of bins may reduce training accuracy but can deal with overfitting.
        "boosting_type": ['gbdt', 'dart'],  # 'gbdt' - traditional Gradient Boosting Decision Tree, 'dart' - Dropouts meet Multiple Additive Regression Trees
        "objective": ["binary"],
        "metric": ["auc", "cross_entropy"],
        "is_unbalance": [True]
    }

    #########################Experiment2####################################
    # Specify the path to your SQLite database
    sheet_path = '../data/Feature_eng_augmented_kyc.csv'
    augmented_data_sheet_df = pd.read_csv(sheet_path)

    y = augmented_data_sheet_df['label']
    to_drop = ["Name", "cust_id", "label"]
    X = augmented_data_sheet_df.drop(to_drop, axis=1)

    # Balancing the dev and test
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.1, random_state=0) #randomly spliting the data to train/test
    X_dev, y_dev, _ , _ = revised_undersample_data(X_dev, y_dev, seed=0)
    X_test, y_test, _ , _ = revised_undersample_data(X_test, y_test, seed=0)

    best_params = grid_search_kfold("LGBMClassifier", lgbm_clf_param_grid, X_dev, y_dev, k)
    best_params["random_state"] = 0
    clf = test_experiment("LGBMClassifier", X_dev, y_dev, X_test, y_test, best_params)
    print("best_params are:", best_params)


    model_filename = '../results/lgbm_model.txt'
    clf.booster_.save_model(model_filename)
    print(f"Model saved to {model_filename}")

    # Load the model from the file
    loaded_model = lgb.Booster(model_file=model_filename)

    # XAI for the LightGBM using shap
    # Calculate SHAP values
    plt.figure()
    explainer = shap.TreeExplainer(loaded_model)
    shap_values = explainer.shap_values(X_dev)

    # Plot summary plot for feature importances
    shap.summary_plot(shap_values, X_dev, plot_type="bar")