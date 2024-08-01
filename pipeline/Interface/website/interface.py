#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2 Created in Dec 2023

Team X
2023-2024 IMI Big Data and Artificial Intelligence Competition
@author: Ernest (Khashayar) Namdar, Tushar Raju
"""

from flask import Flask, render_template, request
import sqlite3
import pandas as pd
from catboost import CatBoostClassifier

app = Flask(__name__)

model = CatBoostClassifier()
model.load_model('website/models/catboost_model.cbm')

@app.route('/', methods=['GET', 'POST'])
def explore():
    if request.method == 'POST':
        cust_id = request.form.get('cust_id')
        
        conn = sqlite3.connect('Scotia.db')
        customer_info, feature_vector = get_customer_info_and_features(conn, cust_id)
        conn.close()

        prediction_proba = model.predict_proba(feature_vector)
        probability_of_positive_class = round(prediction_proba[0][1], 3)


        return render_template('explore_page.html', customer_info=customer_info, prediction=probability_of_positive_class, cust_id=cust_id)

    return render_template('explore_page.html', customer_info=None, prediction=None,  cust_id=None)

@app.route('/',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        gender = request.form.get('gender')
        occupation = request.form.get('occupation')
        age = request.form.get('age')
        tenure = request.form.get('tenure')
        open_sanction = request.form.get('open_sanction')

        gender_mapping = {label: idx for idx, label in enumerate(df['Gender'].astype('category').cat.categories)}
        occupation_mapping = {label: idx for idx, label in enumerate(df['Occupation'].astype('category').cat.categories)}

        gender = gender.map(gender_mapping)
        occupation = occupation.map(occupation_mapping)

        features = pd.DataFrame([gender,occupation,age,tenure,open_sanction])

        prediction_proba = model.predict_proba(features)
        probability_of_positive_class = round(prediction_proba[0][1], 3)

        return render_template('predict_page.html', prediction=probability_of_positive_class)

    return render_template('predict_page.html', prediction=None)

def get_customer_info_and_features(conn, cust_id):

    query = f"SELECT * FROM kyc WHERE cust_id = '{cust_id}'"
    df = pd.read_sql(query, conn)

    customer_info = df.to_dict(orient='records')[0]


    gender_mapping = {label: idx for idx, label in enumerate(df['Gender'].astype('category').cat.categories)}
    occupation_mapping = {label: idx for idx, label in enumerate(df['Occupation'].astype('category').cat.categories)}

    df['Gender'] = df['Gender'].map(gender_mapping)
    df['Occupation'] = df['Occupation'].map(occupation_mapping)

    features = df.drop(["Name", "cust_id", "label"], axis=1)
    return customer_info, features


if __name__ == '__main__':
    app.run(debug=True)
