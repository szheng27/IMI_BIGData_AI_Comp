#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3 Created in Feb 2023

Team X
2023-2024 IMI Big Data and Artificial Intelligence Competition
@author: Steven Zheng, Fiona Li
"""

from fuzzywuzzy import process
import pandas as pd

# Used the function to find the closest-match occupation
def get_closest_match(x, choices):
    return process.extractOne(x, choices, score_cutoff=80)


def main():
    main_dataset = pd.read_csv('../data/kyc.csv')
    occupations_list = pd.read_csv('Occupations_List.csv')

    main_dataset['Closest Occupation Match'] = main_dataset['Occupation'].apply(lambda x: get_closest_match(x, occupations_list['Occupation']))
    main_dataset['Label_Filtered_By_Occupations'] = main_dataset['Closest Occupation Match'].apply(lambda x: 1 if x is not None else 0)

    kyc = pd.read_csv('filtered_kyc.csv')
    name_list = pd.read_excel('suspected_name_list.xlsx')
    merged_suspected_file = pd.merge(kyc, name_list, left_on='cust_id', right_on='id', how='left')
    merged_suspected_file = merged_suspected_file.drop_duplicates(subset=['cust_id'])
    merged_suspected_file['Label_Filtered_By_Wire_EMT'] = merged_suspected_file['id'].apply(lambda x: 1 if pd.notnull(x) else 0)
    merged_suspected_file.drop(['Closest Occupation Match', 'id','Name_y','Gender', 'Occupation', 'Age', 'Tenure'], axis=1, inplace=True)
    type_dummies = pd.get_dummies(merged_suspected_file['Type'], prefix='type')
    merged_suspected_file = pd.concat([merged_suspected_file, type_dummies], axis=1)
    merged_suspected_file.drop('Type', axis=1, inplace=True)
    merged_suspected_file.rename(columns={'Name_x': 'Name', 'label': 'Label_Money_Laundry'}, inplace=True)
    merged_suspected_file.to_csv('merged_suspected_file.csv', index=False)

    # Combine with master file
    new_kyc = pd.read_csv('kyc_full_all_columns.csv')
    final_clustering_file = pd.merge(new_kyc, merged_suspected_file, left_on='cust_id', right_on='cust_id', how='left')
    final_clustering_file.to_csv('final_clustering_file.csv', index=False)
    
    # Combine with clustering result
    final_dataset = pd.read_csv('merged_suspected_file.csv')
    clustering_dataset = pd.read_csv('cluster11.csv')
    merged_df = pd.merge(final_dataset, clustering_dataset[['cust_id']], on='cust_id', how='left', indicator=True)
    merged_df['label by clustering'] = merged_df['_merge'].apply(lambda x: 1 if x == 'both' else 0)
    merged_df.drop(columns=['_merge'], inplace=True)
    merged_df.to_csv('suspicious_name_list.csv', index=False)

    # Dataset for prepared for social network
    transaction_dataset = pd.read_csv('risky_transactions_23.csv')
    transaction_datase = transaction_dataset.drop_duplicates()
    grouped_df = transaction_dataset.groupby(['Source', 'Target']).agg(counts=('Source', 'size'),total_transaction_amount=('Amount', 'sum')).reset_index()
    grouped_df.to_csv('grouped_df_file23.csv', index=False)

    # Updating the weight
    name_dataset = pd.read_csv('52_name_list.csv')
    group_by_dataset = pd.read_csv('grouped_df_file2.csv')
    name_merged_df = pd.merge(name_dataset, group_by_dataset, on='Source', how='left')

    # Transactions within 2 hops for both task 2 and task 3 suspicious name list
    main_df = pd.read_csv('grouped_df_file23.csv')
    secondary_df = pd.read_csv('grouped_df_file2.csv')
    merged_df = pd.merge(main_df, secondary_df, on=['Source', 'Target'], how='left')
    merged_df.to_csv('merged_df1.csv', index=False)



if __name__ == "__main__":
    main()