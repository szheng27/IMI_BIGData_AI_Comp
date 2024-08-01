#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3 Created in Feb 2023

Team X
2023-2024 IMI Big Data and Artificial Intelligence Competition
@author: Tushar Raju
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def flatten_list(row):
    if isinstance(row, list):
        return [str(word) for sublist in row if isinstance(sublist, list) for word in sublist if pd.notna(word)]
    elif pd.notna(row):
        return [str(row)]
    else:
        return []

def main():
    kyc = pd.read_csv("kyc_with_open_sanctions_features.csv")
    wires = pd.read_csv("wire_trxns.csv")
    emts = pd.read_csv("emt_trxns.csv")

    original_categories = kyc['Occupation'].astype('category').cat.categories

    kyc['Gender'] = kyc['Gender'].astype('category').cat.codes
    kyc['Occupation'] = pd.Categorical(kyc['Occupation'], categories=original_categories)

    wildlife_occupations = pd.read_excel("occupations.xlsx",sheet_name="wildlife")
    wildlife_occupations['wildlife'] = pd.Categorical(wildlife_occupations['wildlife'], categories=original_categories)

    trafficking_occupations = pd.read_excel("occupations.xlsx",sheet_name="industry")
    trafficking_occupations['trafficking'] = pd.Categorical(trafficking_occupations['trafficking'], categories=original_categories)

    transportation_occupations = pd.read_excel("occupations.xlsx",sheet_name="transportation")
    transportation_occupations['transportation'] = pd.Categorical(transportation_occupations['transportation'], categories=original_categories)

    media_occupations = pd.read_excel("occupations.xlsx",sheet_name="media")
    media_occupations['media'] = pd.Categorical(media_occupations['media'], categories=original_categories)

    law_occupations = pd.read_excel("occupations.xlsx",sheet_name="law")
    law_occupations['law_enforcement'] = pd.Categorical(law_occupations['law_enforcement'], categories=original_categories)

    wildlife_occupation = wildlife_occupations['wildlife'].tolist()
    trafficking_occupation = trafficking_occupations['trafficking'].tolist()
    transportation_occupation = transportation_occupations['transportation'].tolist()
    media_occupation = media_occupations['media'].tolist()
    law_occupation = law_occupations['law_enforcement'].tolist()


    kyc['wildlife_occupation'] = kyc['Occupation'].isin(wildlife_occupation).astype(int)
    kyc['trafficking_occupation'] = kyc['Occupation'].isin(trafficking_occupation).astype(int)
    kyc['transportation_occupation'] = kyc['Occupation'].isin(transportation_occupation).astype(int)
    kyc['media_occupation'] = kyc['Occupation'].isin(media_occupation).astype(int)
    kyc['law_occupation'] = kyc['Occupation'].isin(law_occupation).astype(int)

    # Aggregating wires data into KYC
    merged_data = pd.merge(wires, kyc[['cust_id', 'Occupation']],   left_on='id sender', right_on='cust_id', how='left')

    merged_data = merged_data.drop(columns=['cust_id'])
    merged_data = merged_data.rename(columns={'Occupation':'Sender Occupation'})
    wires = merged_data

    merged_data = pd.merge(wires, kyc[['cust_id', 'Occupation']],
                        left_on='id receiver', right_on='cust_id', how='left')

    merged_data = merged_data.drop(columns=['cust_id'])
    merged_data = merged_data.rename(columns={'Occupation':'Receiver Occupation'})
    wires = merged_data

    merged_data = pd.merge(kyc, wires[['id sender', 'country receiver', 'id receiver','Receiver Occupation']], left_on='cust_id', right_on='id sender', how='left')

    grouped_data = merged_data.groupby('cust_id').apply(lambda x: list(x['Receiver Occupation'])).reset_index(name='wire_trxn_sent_occupations')

    result_data = pd.merge(kyc, grouped_data, left_on='cust_id', right_on='cust_id', how='left')
    kyc = result_data

    merged_data = pd.merge(kyc, wires[['id sender', 'country receiver', 'id receiver','Receiver Occupation']],
                        left_on='cust_id', right_on='id sender', how='left')

    grouped_data = merged_data.groupby('cust_id').apply(lambda x: list(x['country receiver'])).reset_index(name='wire_trxn_sent_countries')

    result_data = pd.merge(kyc, grouped_data, left_on='cust_id', right_on='cust_id', how='left')
    kyc = result_data

    merged_data = pd.merge(kyc, wires[['id sender', 'country sender', 'id receiver','Sender Occupation']],
                       left_on='cust_id', right_on='id receiver', how='left')

    grouped_data = merged_data.groupby('cust_id').apply(lambda x: list(x['Sender Occupation'])).reset_index(name='wire_trxn_received_occupations')

    result_data = pd.merge(kyc, grouped_data, left_on='cust_id', right_on='cust_id', how='left')
    kyc = result_data

    merged_data = pd.merge(kyc, wires[['id sender', 'country sender', 'id receiver','Sender Occupation']],
                        left_on='cust_id', right_on='id receiver', how='left')

    grouped_data = merged_data.groupby('cust_id').apply(lambda x: list(x['country sender'])).reset_index(name='wire_trxn_received_countries')

    result_data = pd.merge(kyc, grouped_data, left_on='cust_id', right_on='cust_id', how='left')
    kyc = result_data

    merged_data = pd.merge(kyc, wires[['id sender', 'id receiver']], left_on='cust_id', right_on='id receiver', how='left')

    grouped_data = merged_data.groupby('cust_id').apply(lambda x: list(x['id sender'])).reset_index(name='wire_trxn_sent_to')

    result_data = pd.merge(kyc, grouped_data, left_on='cust_id', right_on='cust_id', how='left')
    kyc = result_data

    merged_data = pd.merge(kyc, wires[['id sender','id receiver']],
                        left_on='cust_id', right_on='id sender', how='left')

    grouped_data = merged_data.groupby('cust_id').apply(lambda x: list(x['id receiver'])).reset_index(name='wire_trxn_received_from')

    result_data = pd.merge(kyc, grouped_data, left_on='cust_id', right_on='cust_id', how='left')
    kyc = result_data

    kyc['average_wire_sent'] = kyc['wire_total_amnt_sent']/kyc['no_wire_trxns_sent']
    kyc['average_wire_received'] = kyc['wire_total_amnt_received']/kyc['no_wire_trxns_received']
    kyc['no_external_wire_trxn_received'] = kyc['wire_trxn_sent_to'].apply(lambda x: sum(1 for item in x if isinstance(item, str) and item.startswith('E')))
    kyc['no_external_wire_trxn_sent'] = kyc['wire_trxn_received_from'].apply(lambda x: sum(1 for item in x if isinstance(item, str) and item.startswith('E')))
    kyc['no_wire_wildlife_occupation_sent'] = kyc['wire_trxn_sent_occupations'].apply(lambda x: sum(1 for item in x if item in wildlife_occupation))
    kyc['no_wire_wildlife_occupation_received'] = kyc['wire_trxn_received_occupations'].apply(lambda x: sum(1 for item in x if item in wildlife_occupation))
    kyc['no_wire_trafficking_occupation_sent'] = kyc['wire_trxn_sent_occupations'].apply(lambda x: sum(1 for item in x if item in trafficking_occupation))
    kyc['no_wire_trafficking_occupation_received'] = kyc['wire_trxn_received_occupations'].apply(lambda x: sum(1 for item in x if item in trafficking_occupation))
    kyc['no_wire_transportation_occupation_sent'] = kyc['wire_trxn_sent_occupations'].apply(lambda x: sum(1 for item in x if item in transportation_occupation))
    kyc['no_wire_transportation_occupation_received'] = kyc['wire_trxn_received_occupations'].apply(lambda x: sum(1 for item in x if item in transportation_occupation))
    kyc['no_wire_media_occupation_sent'] = kyc['wire_trxn_sent_occupations'].apply(lambda x: sum(1 for item in x if item in media_occupation))
    kyc['no_wire_media_occupation_received'] = kyc['wire_trxn_received_occupations'].apply(lambda x: sum(1 for item in x if item in media_occupation))
    kyc['no_wire_law_occupation_sent'] = kyc['wire_trxn_sent_occupations'].apply(lambda x: sum(1 for item in x if item in law_occupation))
    kyc['no_wire_law_occupation_received'] = kyc['wire_trxn_received_occupations'].apply(lambda x: sum(1 for item in x if item in law_occupation))
    kyc['no_wire_sent_wildlife_jurisdiction'] = kyc['wire_trxn_sent_countries'].apply(lambda x: sum(1 for item in x if item in ['CN','AU','SA']))
    kyc['no_wire_received_wildlife_jurisdiction'] = kyc['wire_trxn_received_countries'].apply(lambda x: sum(1 for item in x if item in ['CN','AU','SA']))
    kyc['no_wire_sent_north_america'] = kyc['wire_trxn_sent_countries'].apply(lambda x: sum(1 for item in x if item in ['CA']))
    kyc['no_wire_received_north_america'] = kyc['wire_trxn_received_countries'].apply(lambda x: sum(1 for item in x if item in ['CA']))

    # Aggregating EMTs into KYC data

    emts = pd.read_csv("emt_trxns.csv")
    emts['emt message list'] = emts['emtmessage'].apply(lambda x: [word.lower() for word in str(x).split()] if pd.notna(x) else [])

    merged_data = pd.merge(emts, kyc[['cust_id', 'Occupation']],left_on='idsender', right_on='cust_id', how='left')
    merged_data = merged_data.drop(columns=['cust_id'])
    merged_data = merged_data.rename(columns={'Occupation':'Sender Occupation'})
    emts = merged_data

    merged_data = pd.merge(emts, kyc[['cust_id', 'Occupation']],
                       left_on='idreceiver', right_on='cust_id', how='left')

    merged_data = merged_data.drop(columns=['cust_id'])
    merged_data = merged_data.rename(columns={'Occupation':'Receiver Occupation'})
    emts = merged_data

    merged_data = pd.merge(kyc, emts[['idsender', 'emt message list', 'idreceiver','Receiver Occupation']],left_on='cust_id', right_on='idsender', how='left')

    grouped_data = merged_data.groupby('cust_id').apply(lambda x: list(x['Receiver Occupation'])).reset_index(name='emt_trxn_sent_occupations')

    result_data = pd.merge(kyc, grouped_data, left_on='cust_id', right_on='cust_id', how='left')
    kyc = result_data

    merged_data = pd.merge(kyc, emts[['idsender', 'emt message list', 'idreceiver','Sender Occupation']],
                        left_on='cust_id', right_on='idreceiver', how='left')

    grouped_data = merged_data.groupby('cust_id').apply(lambda x: list(x['Sender Occupation'])).reset_index(name='emt_trxn_received_occupations')

    result_data = pd.merge(kyc, grouped_data, left_on='cust_id', right_on='cust_id', how='left')
    kyc = result_data

    merged_data = pd.merge(kyc, emts[['idsender', 'emt message list', 'idreceiver','Receiver Occupation']],left_on='cust_id', right_on='idsender', how='left')

    grouped_data = merged_data.groupby('cust_id').apply(lambda x: list(x['emt message list'])).reset_index(name='emt_trxn_sent_to_msg')

    result_data = pd.merge(kyc, grouped_data, left_on='cust_id', right_on='cust_id', how='left')
    kyc = result_data

    merged_data = pd.merge(kyc, emts[['idsender', 'emt message list', 'idreceiver','Sender Occupation']],
                        left_on='cust_id', right_on='idreceiver', how='left')

    grouped_data = merged_data.groupby('cust_id').apply(lambda x: list(x['emt message list'])).reset_index(name='emt_trxn_received_from_msg')

    result_data = pd.merge(kyc, grouped_data, left_on='cust_id', right_on='cust_id', how='left')
    kyc = result_data

    merged_data = pd.merge(kyc, emts[['idsender', 'idreceiver']],
                       left_on='cust_id', right_on='idreceiver', how='left')

    grouped_data = merged_data.groupby('cust_id').apply(lambda x: list(x['idsender'])).reset_index(name='emt_trxn_sent_to')

    result_data = pd.merge(kyc, grouped_data, left_on='cust_id', right_on='cust_id', how='left')
    kyc = result_data

    merged_data = pd.merge(kyc, emts[['idsender','idreceiver']],
                       left_on='cust_id', right_on='idsender', how='left')

    grouped_data = merged_data.groupby('cust_id').apply(lambda x: list(x['idreceiver'])).reset_index(name='emt_trxn_received_from')

    result_data = pd.merge(kyc, grouped_data, left_on='cust_id', right_on='cust_id', how='left')
    kyc = result_data

    kyc['average_emt_sent'] = kyc['emt_total_amnt_sent']/kyc['no_emt_trxns_sent']
    kyc['average_emt_received'] = kyc['emt_total_amnt_received']/kyc['no_emt_trxns_received']
    kyc['no_external_emt_trxn_received'] = kyc['emt_trxn_sent_to'].apply(lambda x: sum(1 for item in x if isinstance(item, str) and item.startswith('E')))
    kyc['no_external_emt_trxn_sent'] = kyc['emt_trxn_received_from'].apply(lambda x: sum(1 for item in x if isinstance(item, str) and item.startswith('E')))
    kyc['no_emt_wildlife_occupation_sent'] = kyc['emt_trxn_sent_occupations'].apply(lambda x: sum(1 for item in x if item in wildlife_occupation))
    kyc['no_emt_wildlife_occupation_received'] = kyc['emt_trxn_received_occupations'].apply(lambda x: sum(1 for item in x if item in wildlife_occupation))
    kyc['no_emt_trafficking_occupation_sent'] = kyc['emt_trxn_sent_occupations'].apply(lambda x: sum(1 for item in x if item in trafficking_occupation))
    kyc['no_emt_trafficking_occupation_received'] = kyc['emt_trxn_received_occupations'].apply(lambda x: sum(1 for item in x if item in trafficking_occupation))
    kyc['no_emt_transportation_occupation_sent'] = kyc['emt_trxn_sent_occupations'].apply(lambda x: sum(1 for item in x if item in transportation_occupation))
    kyc['no_emt_transportation_occupation_received'] = kyc['emt_trxn_received_occupations'].apply(lambda x: sum(1 for item in x if item in transportation_occupation))
    kyc['no_emt_media_occupation_sent'] = kyc['emt_trxn_sent_occupations'].apply(lambda x: sum(1 for item in x if item in media_occupation))
    kyc['no_emt_media_occupation_received'] = kyc['emt_trxn_received_occupations'].apply(lambda x: sum(1 for item in x if item in media_occupation))
    kyc['no_emt_law_occupation_sent'] = kyc['emt_trxn_sent_occupations'].apply(lambda x: sum(1 for item in x if item in law_occupation))
    kyc['no_emt_law_occupation_received'] = kyc['emt_trxn_received_occupations'].apply(lambda x: sum(1 for item in x if item in law_occupation))

    kyc['emt_sent_messages'] = kyc['emt_trxn_sent_to_msg'].apply(flatten_list)
    kyc['emt_received_messages'] = kyc['emt_trxn_received_from_msg'].apply(flatten_list)

    wildlife_words = pd.read_excel("unique_words.xlsx",sheet_name="wildlife")
    transportation_words = pd.read_excel("unique_words.xlsx",sheet_name="transportation")
    law_words = pd.read_excel("unique_words.xlsx",sheet_name="law")
    trafficking_words = pd.read_excel("unique_words.xlsx",sheet_name="trafficking")
    accomodation_words = pd.read_excel("unique_words.xlsx",sheet_name="accomodation")

    wildlife_word = wildlife_words['wildlife'].tolist()
    transportation_word = transportation_words['transportation'].tolist()
    law_word = law_words['law'].tolist()
    trafficking_word = trafficking_words['trafficking'].tolist()
    accomodation_word = accomodation_words['accomodation'].tolist()

    kyc['no_emt_wildlife_msg_sent'] = kyc['emt_sent_messages'].apply(lambda x: sum(1 for item in x if item in wildlife_word))
    kyc['no_emt_wildlife_msg_received'] = kyc['emt_received_messages'].apply(lambda x: sum(1 for item in x if item in wildlife_word))
    kyc['no_emt_transportation_msg_sent'] = kyc['emt_sent_messages'].apply(lambda x: sum(1 for item in x if item in transportation_word))
    kyc['no_emt_transportation_msg_received'] = kyc['emt_received_messages'].apply(lambda x: sum(1 for item in x if item in transportation_word))
    kyc['no_emt_law_msg_sent'] = kyc['emt_sent_messages'].apply(lambda x: sum(1 for item in x if item in law_word))
    kyc['no_emt_law_msg_received'] = kyc['emt_received_messages'].apply(lambda x: sum(1 for item in x if item in law_word))
    kyc['no_emt_trafficking_msg_sent'] = kyc['emt_sent_messages'].apply(lambda x: sum(1 for item in x if item in trafficking_word))
    kyc['no_emt_trafficking_msg_received'] = kyc['emt_received_messages'].apply(lambda x: sum(1 for item in x if item in trafficking_word))
    kyc['no_emt_accomodation_msg_sent'] = kyc['emt_sent_messages'].apply(lambda x: sum(1 for item in x if item in accomodation_word))
    kyc['no_emt_accomodation_msg_received'] = kyc['emt_received_messages'].apply(lambda x: sum(1 for item in x if item in accomodation_word))

    Names = kyc['Name']
    Occupations = kyc['Occupation']
    kyc = kyc.drop(columns=['Name','Occupation'])
    kyc.fillna(0,inplace=True)

    kyc.to_csv('kyc_full_all_columns.csv', index=False)

if __name__ == "__main__":
    main()