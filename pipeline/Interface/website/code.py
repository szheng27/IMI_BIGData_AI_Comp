#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2 Created in Dec 2023

Team X
2023-2024 IMI Big Data and Artificial Intelligence Competition
@author: Ernest (Khashayar) Namdar, Tushar Raju
"""

from flask import Flask, render_template, request, Blueprint
import sqlite3
import pandas as pd
from catboost import CatBoostClassifier
import numpy as np
import requests
from bs4 import BeautifulSoup
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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

code = Blueprint('code',__name__)

model_cat = CatBoostClassifier()
model_cat.load_model('website/models/catboost_model.cbm')
llm_model = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
nlp = pipeline('question-answering', model=llm_model, tokenizer=llm_model)

@code.route('/')
def landing():
    return render_template("landing_page.html")

@code.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        name = request.form.get('name')
        gender = request.form.get('gender')
        occupation = request.form.get('occupation')
        age = request.form.get('age')
        tenure = request.form.get('tenure')
        open_sanction = request.form.get('open_sanction')

        index = [1, 2, 3, 4, 5]
        features = pd.DataFrame({'Gender': gender.lower(), 'Occupation': occupation, 'Age': age, 'Tenure': tenure,'OpenSanc_query':open_sanction},index=index)

        gender_mapping = {label: idx for idx, label in enumerate(features['Gender'].astype('category').cat.categories)}
        occupation_mapping = {label: idx for idx, label in enumerate(features['Occupation'].astype('category').cat.categories)}

        features['Gender'] = features['Gender'].map(gender_mapping)
        features['Occupation'] = features['Occupation'].map(occupation_mapping)

        prediction_proba = model_cat.predict_proba(features)
        probability_of_positive_class = round(prediction_proba[0][1], 3)

        prediction = np.round(probability_of_positive_class*100,2)

        risk_color = 0
        if prediction < 30:
            risk_color = 1
        elif prediction < 70:
            risk_color = 2
        else:
            risk_color = 3

        if open_sanction == '1':
            os_query = 'Yes'
        else:
            os_query = 'No'

        return render_template('predict_page.html', prediction=prediction,risk_color=risk_color,name=name, gender=gender,occupation=occupation,age=age,tenure=tenure,open_sanction=os_query)

    return render_template('predict_page.html', prediction=None,risk_color=None,name=None, gender=None,occupation=None,age=None,tenure=None,open_sanction=None)

@code.route('/explore',methods=['GET','POST'])
def explore():
    if request.method == 'POST':
        cust_id = request.form.get('cust_id')
        conn = sqlite3.connect('Scotia.db')
        customer_info, feature_vector = get_customer_info_and_features(conn, cust_id)
        conn.close()
        if customer_info is not None:
            prediction_proba = model_cat.predict_proba(feature_vector)
            probability_of_positive_class = round(prediction_proba[0][1], 3)

            name = customer_info['Name'].title()
            gender = customer_info['Gender'].title()
            occupation = customer_info['Occupation'].title()
            age = int(customer_info['Age'])
            tenure = int(customer_info['Tenure'])
            open_sanction = int(customer_info['OpenSanc_query'])
            open_sanction_label = 'No'
            if open_sanction == 1:
                open_sanction_label = 'Yes'
            label = 'No'
            if customer_info['label'] == 1:
                label = 'Yes'
            prediction = np.round(probability_of_positive_class*100,2)

            risk_color = 0
            if prediction < 30:
                risk_color = 1
            elif prediction < 70:
                risk_color = 2
            else:
                risk_color = 3
    
            return render_template('explore_page.html', name=name, gender=gender,occupation=occupation,age=age,tenure=tenure,label=label,  prediction=prediction, cust_id=cust_id,risk_color=risk_color,open_sanction=open_sanction_label)
    return render_template('explore_page.html', name=None, gender=None,occupation=None,age=None,tenure=None,label=None , prediction=None, cust_id=None, risk_color=None,open_sanction=None)

def get_customer_info_and_features(conn, cust_id):
    try:
        query = f"SELECT * FROM kyc WHERE cust_id = '{cust_id}'"
        df = pd.read_sql(query, conn)

        customer_info = df.to_dict(orient='records')[0]


        gender_mapping = {label: idx for idx, label in enumerate(df['Gender'].astype('category').cat.categories)}
        occupation_mapping = {label: idx for idx, label in enumerate(df['Occupation'].astype('category').cat.categories)}

        df['Gender'] = df['Gender'].map(gender_mapping)
        df['Occupation'] = df['Occupation'].map(occupation_mapping)

        features = df.drop(["Name", "cust_id", "label"], axis=1)
    except:
        customer_info = None
        features = None
    return customer_info, features

@code.route('/about')
def about():
    return render_template('about_page.html')

@code.route('/contact')
def contact():
    return render_template('contact_page.html')

@code.route('/find',methods=['GET','POST'])
def find():
    if request.method == 'POST':
        if 'search_btn' in request.form:
            name = request.form.get('os_name')
            search_results = search_open_sanctions(name)
            if search_results == 0:
                result = "No"
            elif search_results == 1:
                result = "Yes"
            else:  
                result = "NA"
            return render_template('find_trafficker_page.html', name=name, open_sanction=result,t_3=None,link=None)
        if 'link_btn' in request.form:
            t_3_name = request.form.get('t_3')
            conn = sqlite3.connect('Scotia.db')
            try:
                criminal_info, feature_vector = get_criminal_info(conn, t_3_name)
                conn.close()
            except:
                conn.close()
                return render_template('find_trafficker_page.html', name=None, open_sanction=None,t_3=None,link=None)
            t_3 = t_3_name
            link = criminal_info['Links']
            return render_template('find_trafficker_page.html', name=None, open_sanction=None,t_3=t_3,link=link)
    return render_template('find_trafficker_page.html',name=None,open_sanction=None,t_3=None,link=None)

def search_open_sanctions(name):
    name_reformed = name.replace(" ", "+")
    api_url = "https://www.opensanctions.org/search/?q="+name_reformed
    response = requests.get(api_url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        alert_div = soup.find("div", class_="alert-heading")
        
        if alert_div and "No matching entities were found" in alert_div.text:
            return 0
        else:
            return 1
    else:
        return -1
    
def get_criminal_info(conn, name):
    try:
        query = f"SELECT * FROM found_criminals WHERE Criminal = '{name}'"
        df = pd.read_sql(query, conn)

        criminal_info = df.to_dict(orient='records')[0]
        features = df.drop(["Name", "cust_id", "label"], axis=1)
    except:
        customer_info = None
        features = None
    return criminal_info, features

def plot_roc_curve(y_true, y_proba, filename='roc_curve.png'):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(filename)

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''
    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar,
                xticklabels=categories if xyticks else False,
                yticklabels=categories if xyticks else False)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
    plt.savefig('confusion_matrix.png')

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
    predictions = clf.predict_proba(X_test)[:, 1]
    return clf, roc_auc_score(y_test, predictions)

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

def test_experiment(model_name, X_dev, y_dev, X_test, y_test, hypers):
    clf = eval(model_name)(**hypers)
    clf.fit(X_dev, y_dev)
    predictions_proba = clf.predict_proba(X_test)[:, 1]  # Probability scores for the positive class
    predictions = clf.predict(X_test)  # Actual predictions
    roc_auc = roc_auc_score(y_test, predictions_proba)  # Calculate ROC AUC score
    return clf, roc_auc, predictions, predictions_proba

def CML(model_name_tag):
    k = 10  # Number of folds

    lgbm_clf_param_grid = {
        "n_estimators": [50, 100, 200], 
        "learning_rate": [0.01, 0.1, 0.2], 
        "max_depth": [5, -1],  # Maximum tree depth for base learners, -1 means no limit
        # LightGBM does not use 'depth' parameter as in CatBoost. Instead, 'max_depth' is used for control over tree depth.
        # "reg_lambda": [0, 1, 10],  # L2 regularization term on weights, equivalent to 'l2_leaf_reg' in CatBoost
        # "max_bin": [255, 510],  # Number of bins that feature values will be bucketed in. Small number of bins may reduce training accuracy but can deal with overfitting.
        # "boosting_type": ['gbdt', 'dart'],  # 'gbdt' - traditional Gradient Boosting Decision Tree, 'dart' - Dropouts meet Multiple Additive Regression Trees
        "objective": ["binary"],
        "metric": ["auc"],
        "is_unbalance": [True]
    }

    #########################Experiment2####################################
    # Specify the path to your SQLite database
    database_path = 'Scotia.db'

    # Create a connection to the SQLite database
    conn = sqlite3.connect(database_path)

    # SQL query to select everything from the view
    query = "SELECT * FROM kyc"
    augmented_data_sheet_df = pd.read_sql_query(query, conn)
    # Close the connection
    conn.close()

    # One-Hot Encoding for Gender
    Gender_dummies = pd.get_dummies(augmented_data_sheet_df['Gender'], prefix='Gender')
    augmented_data_sheet_df = pd.concat([augmented_data_sheet_df, Gender_dummies], axis=1)
    augmented_data_sheet_df.drop('Gender', axis=1, inplace=True)

    # One-Hot Encoding for Occupation
    occupation_dummies = pd.get_dummies(augmented_data_sheet_df['Occupation'], prefix='Occupation')
    augmented_data_sheet_df = pd.concat([augmented_data_sheet_df, occupation_dummies], axis=1)
    augmented_data_sheet_df.drop('Occupation', axis=1, inplace=True)
    augmented_data_sheet_df = augmented_data_sheet_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    y = augmented_data_sheet_df['label']
    to_drop = ["Name", "cust_id", "label"]
    X = augmented_data_sheet_df.drop(to_drop, axis=1)

    # Balancing the dev and test
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.1, random_state=0) #randomly spliting the data to train/test
    X_dev, y_dev, _ , _ = revised_undersample_data(X_dev, y_dev, seed=0)
    X_test, y_test, _ , _ = revised_undersample_data(X_test, y_test, seed=0)

    best_params = grid_search_kfold("LGBMClassifier", lgbm_clf_param_grid, X_dev, y_dev, k)
    best_params["random_state"] = 0

    clf, test_performance, predictions, predictions_proba = test_experiment("LGBMClassifier", X_dev, y_dev, X_test, y_test, best_params)

    plot_roc_curve(y_test, predictions_proba)
    cf = confusion_matrix(y_test, predictions)

    print(cf)

    make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None)

    model_filename = 'lgbm_model_'+model_name_tag+'.txt'
    clf.booster_.save_model(model_filename)

    return best_params, test_performance

@code.route('/model', methods=['GET','POST'])
def model():
    if request.method == 'POST':
        model_name = request.form.get('model_name')
        if model_name is not None:
            best_params, test_performance = CML(model_name)
            return render_template('model_page.html', model_name=model_name)
    return render_template('model_page.html', model_name=None)

@code.route('/news', methods=['GET','POST'])
def news():
    criminal = None
    if request.method == 'POST':
        news_text = request.form.get('news_text')
        if news_text is not None:
            criminal = identify_criminal(news_text)
            return render_template('news_page.html', criminal=criminal)
    return render_template('news_page.html', criminal=None)
    
def identify_criminal(text):
     q1 = "Who is the criminal?"
     response1 = nlp({
            'question': q1,
            'context': text
        })
     return response1['answer']