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
import string
import csv

def drop_words(source_list, words_to_drop):
    result_list = [word for word in source_list if word not in words_to_drop]
    return result_list

def main():
    kyc_data = pd.read_csv("../FeatureEngineering2.csv")

    # Creating Bag of Words for EMT Transactions

    emts = pd.read_csv("../emt_trxns.csv")
    messages = emts['emt message'].dropna().tolist()

    translator = str.maketrans("", "", string.punctuation)
    tokenized_sentences = [message.translate(translator).lower().split() for message in messages]


    unique_words = list(set(word.strip() for words in tokenized_sentences for word in words))
    len(unique_words)

    #Extracting Bag of Words

    csv_file_path = 'unique_words.csv'

    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows([[word] for word in unique_words])

    word_bag = pd.read_csv("unique_words.csv")

    somewhat_related_words = word_bag['somewhat_related_trafficking_or_animals']
    strongly_related_words = word_bag['strongly_related_trafficking_or_animal']
    strongly_related_words = strongly_related_words.dropna().tolist()

    somewhat_related_words = drop_words(somewhat_related_words, strongly_related_words)
    
if __name__ == "__main__":
    main()