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
import matplotlib as mpl
import seaborn as sns


if __name__ == "__main__":
    data_sheet_path = "../data/kyc.csv" # know your customer sheet that contains the GT labels
    data_sheet_df = pd.read_csv(data_sheet_path)

    # Data Exploration#########################################################

    # Pie Chart
    # Count the occurrences of each unique label
    label_counts = data_sheet_df['label'].value_counts()
    
    # Define a new color-blind-friendly color palette
    colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9']
    
    # Increase font size
    mpl.rcParams['font.size'] = 12
    
    # Plotting the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(label_counts, labels=label_counts.index, autopct=lambda p: '{:.0f}\n({:.1f}%)'.format(p * sum(label_counts) / 100, p), 
            startangle=140, colors=colors)
    plt.title('Distribution of Labels')
    plt.show()
    ##############################################################
    # Unique Values
    # Iterate over each column to get unique values and counts
    for column in data_sheet_df.columns:
        unique_values = data_sheet_df[column].unique()
        unique_count = data_sheet_df[column].nunique()
        print(f"Column: {column}")
        print(f"Unique Values: {unique_values}")
        print(f"Count of Unique Values: {unique_count}")
        print("-" * 50)
    ##############################################################
    # label-colored plots
    ###########   AGE    ###############
    # Determine the unique labels
    unique_labels = data_sheet_df['label'].unique()
    
    # Determine the common bin range for all subsets
    min_age = data_sheet_df['Age'].min()
    max_age = data_sheet_df['Age'].max()
    bins = range(int(min_age), int(max_age) + 2, 2) # Adjust the bin width as needed
    
    # Setting up a clean and simple plot style
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(10, 6))
    
    # Plot histograms for each label with consistent bins
    for label in unique_labels:
        subset = data_sheet_df[data_sheet_df['label'] == label]
        plt.hist(subset['Age'], bins=bins, alpha=0.7, label=str(label), edgecolor='black')
    
    # Enhancing clarity and simplicity
    plt.title('Age Distribution by Label', fontsize=14, weight='bold')
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Label', frameon=False)
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    # Minimizing chartjunk
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    
    plt.show()
    ###########   Tenure  ###############
    # Determine the common bin range for all subsets
    min_Tenure = data_sheet_df['Tenure'].min()
    max_Tenure = data_sheet_df['Tenure'].max()
    bins = range(int(min_Tenure), int(max_Tenure) + 2, 2) # Adjust the bin width as needed
    
    # Setting up a clean and simple plot style
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(10, 6))
    
    # Plot histograms for each label with consistent bins
    for label in unique_labels:
        subset = data_sheet_df[data_sheet_df['label'] == label]
        plt.hist(subset['Tenure'], bins=bins, alpha=0.7, label=str(label), edgecolor='black')
    
    # Enhancing clarity and simplicity
    plt.title('Tenure Distribution by Label', fontsize=14, weight='bold')
    plt.xlabel('Tenure', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Label', frameon=False)
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    # Minimizing chartjunk
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    
    plt.show()
    ##############  Gender ###########################
    # Create a summary table with counts of each gender for each label
    summary_table = data_sheet_df.groupby(['Gender', 'label']).size().unstack(fill_value=0)
    
    # Calculate the total counts by gender for percentage calculation
    total_counts_by_gender = summary_table.sum(axis=1)
    
    # Convert counts to percentages
    summary_table_percentage = summary_table.div(total_counts_by_gender, axis=0) * 100
    
    # Setting up the Seaborn style
    sns.set(style="whitegrid")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Define the number of unique genders and labels for plotting
    n_genders = len(summary_table.index)
    n_labels = len(summary_table.columns)
    bar_width = 0.35
    
    # Create bars for each gender-label combination
    for i, label in enumerate(summary_table.columns):
        bars = plt.bar(np.arange(n_genders) + i * bar_width, summary_table[label], width=bar_width, label=str(label))
        
        # Add percentage annotations
        for bar, percent in zip(bars, summary_table_percentage[label]):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f'{percent:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Styling and labeling the plot
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Gender Distribution by Label', fontsize=14, weight='bold')
    plt.xticks(np.arange(n_genders) + bar_width * (n_labels - 1) / 2, summary_table.index)
    plt.legend(title='Label', frameon=False)
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    # Minimizing chartjunk
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    
    plt.show()
    ################ Occupation #####################
    # Create a summary table with counts of each occupation for each label
    summary_table = data_sheet_df.groupby(['Occupation', 'label']).size().unstack(fill_value=0)
    
    # Calculate the total counts by occupation for percentage calculation
    total_counts_by_occupation = summary_table.sum(axis=1)
    
    # Convert counts to percentages and sort by the smaller label class
    summary_table_percentage = summary_table.div(total_counts_by_occupation, axis=0) * 100
    sorted_occupations = summary_table_percentage.min(axis=1).sort_values(ascending=False).head(10).index
    
    # Setting up the Seaborn style
    sns.set_theme(style="whitegrid")
    
    # Plotting
    plt.figure(figsize=(12, 10)) # Adjusted for top 10 occupations
    
    # Define the number of unique occupations (top 10) and labels for plotting
    n_occupations = len(sorted_occupations)
    n_labels = len(summary_table.columns)
    bar_height = 0.4
    
    # Create horizontal bars for each top 10 occupation-label combination
    for i, label in enumerate(summary_table.columns):
        # Sort and select top 10 occupations
        sorted_values = summary_table.loc[sorted_occupations, label]
        bars = plt.barh(np.arange(n_occupations) + i * bar_height, sorted_values, height=bar_height, label=str(label))
    
        # Add count annotations
        for j, bar in enumerate(bars):
            occupation = sorted_occupations[j]
            percentage = summary_table_percentage.loc[occupation, label]
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{percentage:.1f}%', va='center', fontsize=10)
    
    # Styling and labeling the plot
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Occupation', fontsize=12)
    plt.title('Distribution by Label for Top 10 Occupations with Highest Number of Positive Cases', fontsize=14, weight='bold')
    plt.yticks(np.arange(n_occupations) + bar_height * (n_labels - 1) / 2, sorted_occupations)
    plt.gca().invert_yaxis() # Invert y-axis to have the first occupation on top
    plt.legend(title='Label', frameon=False)
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    # Minimizing chartjunk
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.show()
    ##############################################################
    # Identify repeated names
    name_counts = data_sheet_df['Name'].value_counts()
    repeated_names = name_counts[name_counts > 1].index.tolist()
    
    # Filter the DataFrame to include only customers with repeated names
    repeated_names_df = data_sheet_df[data_sheet_df['Name'].isin(repeated_names)]
    
    # Count the occurrences of each unique label among customers with repeated names
    label_counts = repeated_names_df['label'].value_counts()
    
    # Define a new color-blind-friendly color palette
    colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9']
    
    # Increase font size
    mpl.rcParams['font.size'] = 12
    
    # Plotting the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(label_counts, labels=label_counts.index, autopct=lambda p: '{:.0f}\n({:.1f}%)'.format(p * sum(label_counts) / 100, p), 
            startangle=140, colors=colors)
    plt.title('Distribution of Labels for Customers with Repeated Names')
    plt.show()