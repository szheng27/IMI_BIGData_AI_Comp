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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from cf_matrix import make_confusion_matrix
from sklearn.metrics import roc_auc_score


def supervised_model_training(X, y, clf):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0) #randomly spliting the data to train/test
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    cm = confusion_matrix(y_test, predictions)
    make_confusion_matrix(cm, group_names=['0', '1'])
    plt.show()
    predictions = clf.predict_proba(X_test)[:, 1]
    print("AUC score:", roc_auc_score(y_test, predictions))

    return clf


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

    clf = DecisionTreeClassifier(random_state=0)
    clf = supervised_model_training(X, y, clf) #supervised learning over the whole dataset
