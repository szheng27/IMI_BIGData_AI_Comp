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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from cf_matrix import make_confusion_matrix
from sklearn.metrics import roc_auc_score


def balance_dset(df, target):
    high_risk_data = df[df[target]==1].copy()
    labels = df[target].copy()
    all_low_risk = labels[labels == 0]
    low_risk_to_keep = np.random.choice(all_low_risk.index, size=high_risk_data.shape[0], replace=False)
    low_risk_data = df.iloc[low_risk_to_keep].copy()
    new_df = pd.concat([high_risk_data, low_risk_data], axis=0)
    return new_df


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

    #########################Balanced Dataset####################################
    balanced_risk_sheet_df = balance_dset(data_sheet_df, "label") # make the dataset balanced

    lbl_counts = pd.DataFrame(balanced_risk_sheet_df['label'].value_counts())
    # whole dataset class contribution
    labels = list(lbl_counts.index)
    values = [int(value) for value in lbl_counts.values]
    pie = plt.figure()
    plt.pie(values, labels=labels, autopct=lambda p:f'{p:.2f}%, {p*sum(values)/100 :.0f}')
    plt.title("Class Contributions Over the Balanced Dataset")
    plt.axis('off')

    y = balanced_risk_sheet_df['label']
    to_drop = ["Name", "cust_id", "label"]
    X = balanced_risk_sheet_df.drop(to_drop, axis=1)

    clf = DecisionTreeClassifier(random_state=0)
    clf = supervised_model_training(X, y, clf) #supervised learning over the whole dataset