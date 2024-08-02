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


def batch_inference(clf, X_test, y_test):
    prediction_scores = clf.predict(X_test)
    predictions_labels = np.where(prediction_scores > 0.5, 1, 0)
    cm = confusion_matrix(y_test, predictions_labels)
    make_confusion_matrix(cm, group_names=['0', '1'])
    plt.show()
    print("AUC score:", roc_auc_score(y_test, prediction_scores))
    return prediction_scores



if __name__ == "__main__":

    #########################Experiment2####################################
    # Specify the path to your SQLite database
    database_path = '../data/Scotiabank.sqlite'

    # Create a connection to the SQLite database
    conn = sqlite3.connect(database_path)

    # SQL query to select everything from the view
    query = "SELECT * FROM kyc_features"
    augmented_data_sheet_df = pd.read_sql_query(query, conn)
    # Close the connection
    conn.close()
    print(augmented_data_sheet_df.head())

    # One-Hot Encoding for Gender
    Gender_dummies = pd.get_dummies(augmented_data_sheet_df['Gender'], prefix='Gender')
    augmented_data_sheet_df = pd.concat([augmented_data_sheet_df, Gender_dummies], axis=1)
    augmented_data_sheet_df.drop('Gender', axis=1, inplace=True)

    # One-Hot Encoding for Occupation
    occupation_dummies = pd.get_dummies(augmented_data_sheet_df['Occupation'], prefix='Occupation')
    augmented_data_sheet_df = pd.concat([augmented_data_sheet_df, occupation_dummies], axis=1)
    augmented_data_sheet_df.drop('Occupation', axis=1, inplace=True)
    augmented_data_sheet_df = augmented_data_sheet_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    model_filename = '../models/lgbm_model.txt'
    # Load the model from the file
    loaded_model = lgb.Booster(model_file=model_filename)

    #loading the customer ids for inference
    list_ids_inference = '../data/risky_individuals_task3_label_filter.csv'
    ids_inference = pd.read_csv(list_ids_inference)
    batch_inference_df = augmented_data_sheet_df[augmented_data_sheet_df['cust_id'].isin(ids_inference['cust_id'])].reset_index(drop=True)
    
    y = batch_inference_df['label']
    to_drop = ["Name", "cust_id", "label"]
    X = batch_inference_df.drop(to_drop, axis=1)
    predictions = batch_inference(loaded_model, X, y)

    batch_inference_df['predictions'] = predictions

    csv_file_path = '../results/batch_inference.csv'
    
    # Save the DataFrame with predictions to a CSV file
    batch_inference_df.to_csv(csv_file_path, index=False)