#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3 Created in Feb 2023

Team X
2023-2024 IMI Big Data and Artificial Intelligence Competition
@author: Steven Zheng, Fiona Li
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn import datasets
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import matplotlib.cm as cm


def main():
    df_cluster = pd.read_csv('final_clustering_file.csv')
    #select_col = ['Label_Filtered_By_Wire_EMT','Label_Filtered_By_Occupations','no_wire_trxns_sent','wire_total_amnt_sent','wire_total_amnt_received','no_wire_trxns_received','no_emt_trxns_sent','no_emt_trxns_received','emt_total_amnt_sent','emt_total_amnt_received','total_cash_deposit','no_cash_withdraws','no_cash_deposits','total_cash_deposit']
    select_col = ['no_wire_trxns_sent','wire_total_amnt_sent','wire_total_amnt_received','no_wire_trxns_received','no_emt_trxns_sent','no_emt_trxns_received','emt_total_amnt_sent','emt_total_amnt_received','total_cash_deposit','no_cash_withdraws','no_cash_deposits','total_cash_deposit']
    X = df_cluster[select_col]

    # Test
    np.random.seed(42)
    n_samples = 100
    counts = np.random.randint(1, 100, n_samples)
    amounts = np.random.uniform(10, 1000, n_samples)
    binary_features = np.random.choice([0, 1], size=(n_samples, 5))

    # Combine features into a DataFrame
    data = pd.DataFrame({
    'Counts': counts,
    'Amounts': amounts,
    'BinaryFeature1': binary_features[:, 0],
    'BinaryFeature2': binary_features[:, 1],
    'BinaryFeature3': binary_features[:, 2],
    'BinaryFeature4': binary_features[:, 3],
    'BinaryFeature5': binary_features[:, 4]
    })

    np.random.seed(42)
    n_samples = 100
    counts = np.random.randint(1, 100, n_samples)
    amounts = np.random.uniform(10, 1000, n_samples)
    binary_features = np.random.choice([0, 1], size=(n_samples, 5))


    data = pd.DataFrame({
    'Counts': counts,
    'Amounts': amounts,
    'BinaryFeature1': binary_features[:, 0],
    'BinaryFeature2': binary_features[:, 1],
    'BinaryFeature3': binary_features[:, 2],
    'BinaryFeature4': binary_features[:, 3],
    'BinaryFeature5': binary_features[:, 4]
    })


    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(data_scaled)


    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.8)
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

    # Apply t-SNE to dataset, reduce the transactional behavior features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(X)

    #Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(data_scaled)

    #scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.8)
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

    tsne_df = pd.DataFrame(tsne_result, columns=['dr1', 'dr2'])

    # Concatinating results back to dataset
    df_cluster = pd.concat([df_cluster, tsne_df], axis=1)
    df_cluster[['dr1','dr2']]

    # Clustering
    #X_new = df_cluster[['Age','dr1','dr2','wildlife_msgs','wildlife_occupation']]
    X_new = df_cluster[['dr1','dr2','wildlife_msgs','Label_Filtered_By_Occupations']]

    X_new_scale = scaler.fit_transform(X_new)

    df_cluster.corrwith(df_cluster['Label_Filtered_By_Wire_EMT']).sort_values(ascending=False).head(20)

    #K-Means Clustering - Faster Option
    inertias = []
    np.random.seed(1)
    ks = range(2,20)
    for k in ks:
        model = KMeans(n_clusters=k, n_init=10)
        model.fit(X_new_scale)
        inertias.append(model.inertia_)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
    x=list(ks),
    y=inertias,
    mode='lines+markers',
    marker=dict(size=8),
    line=dict(color='blue'),
    ))

    fig.update_layout(
    title='Elbow Method for Optimal Number of Clusters',
    xaxis=dict(title='Number of Clusters'),
    yaxis=dict(title='Inertia'),
    showlegend=False,
    )

    fig.show()

    np.random.seed(1)
    kmeans = KMeans(n_clusters=12, n_init='auto')
    kmeans.fit(X_new_scale)
    labels = kmeans.labels_

    df_cluster['Clusters'] = labels

    cluster_counts = df_cluster["Clusters"].value_counts()
    
    feature_avgs = df_cluster.groupby('Clusters').mean()

    # Exporting results
    df_cluster[df_cluster['Clusters']==11].to_csv('cluster11.csv')

if __name__ == "__main__":
    main()