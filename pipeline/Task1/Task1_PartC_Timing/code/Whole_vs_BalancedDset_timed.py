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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import time
from numpy import percentile
from boxplots import plot_boxes


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


if __name__ == "__main__":
    data_sheet_path = "../data/kyc.csv" # know your customer sheet that contains the GT labels
    data_sheet_df = pd.read_csv(data_sheet_path)

    ##############################################################
    # Encoding
    # Label Encoding for Gender
    data_sheet_df['Gender'] = data_sheet_df['Gender'].astype('category').cat.codes

    # Label Encoding for Occupation
    data_sheet_df['Occupation'] = data_sheet_df['Occupation'].astype('category').cat.codes

    # #########################Whole Dataset#######################################
    # creating the GT vector and the feature matrix
    y = data_sheet_df['label']
    to_drop = ["Name", "cust_id", "label"]
    X = data_sheet_df.drop(to_drop, axis=1)

    start = time.time()
    print("running DT on Whole Dataset")
    N = 30 #Number of experiments
    wholedset_aucs = []
    for i in range(N):
        clf = DecisionTreeClassifier(random_state=i)
        clf, auc = supervised_model_training(X, y, clf, seed=i)
        wholedset_aucs.append(auc)
    min_wholedset, q1_wholedset, med_wholedset, q3_wholedset, max_wholedset = five_number_summary(wholedset_aucs)
    mean_wholedset = np.mean(wholedset_aucs)
    sd_wholedset = np.std(wholedset_aucs)
    print("Wholedset stats (mean, sd, 5stats):", [mean_wholedset, sd_wholedset, min_wholedset, q1_wholedset, med_wholedset, q3_wholedset, max_wholedset])

    end = time.time()
    duration = end-start
    print("The run was completed in: ", int(duration/60), "minutes and ", int(duration%60), "seconds")

    #########################Balanced Dataset####################################
    balanced_risk_sheet_df = balance_dset(data_sheet_df, "label") # make the dataset balanced

    y = balanced_risk_sheet_df['label']
    to_drop = ["Name", "cust_id", "label"]
    X = balanced_risk_sheet_df.drop(to_drop, axis=1)

    start = time.time()
    print("running DT on balanced Dataset")
    N = 30 #Number of experiments
    balanced_aucs = []
    for i in range(N):
        clf = DecisionTreeClassifier(random_state=i)
        clf, auc = supervised_model_training(X, y, clf, seed=i)
        balanced_aucs.append(auc)
    min_balanced, q1_balanced, med_balanced, q3_balanced, max_balanced = five_number_summary(balanced_aucs)
    mean_balanced = np.mean(balanced_aucs)
    sd_balanced = np.std(balanced_aucs)
    print("balanced stats (mean, sd, 5stats):", [mean_balanced, sd_balanced, min_balanced, q1_balanced, med_balanced, q3_balanced, max_balanced])
    end = time.time()
    duration = end-start
    print("The run was completed in: ", int(duration/60), "minutes and ", int(duration%60), "seconds")

    plot_boxes(mean_wholedset, sd_wholedset, [min_wholedset, q1_wholedset, med_wholedset, q3_wholedset, max_wholedset],
               mean_balanced, sd_balanced, [min_balanced, q1_balanced, med_balanced, q3_balanced, max_balanced],
               ["WholeDataset", "BalancedDataset"])