#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3 Created in Feb 2023

Team X
2023-2024 IMI Big Data and Artificial Intelligence Competition
@author: Tushar Raju, Amelyn Wang
"""

import time
import pandas as pd 
import numpy as np
from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
import ast
import spacy 
from collections import defaultdict
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


def load_data():
    # Load scraped data and best_doc
    scraped_data = pd.read_csv("../results/inks_and_titles.csv")
    with open("best_doc_2.txt", 'r') as file:
        best_doc = file.read()
    return scraped_data, best_doc

def clean_data(scraped_data):
    # Clean the scraped data: remove links with weird syntax
    problem_ID, ID, Link, Title, Text = [], [], [], [], []
    for index, row in scraped_data.iterrows():
        title = row['Extracted Data']
        try:
            dict = ast.literal_eval(title)
            Title.append(dict['Title'])
            Text.append(dict['Text'])
            ID.append(index)
            Link.append(row['Link'])
        except SyntaxError: # string representation error, no " at the end
            problem_ID.append(index)
        except TypeError: # Input is not a string
            problem_ID.append(index)
    length = len(scraped_data[['Extracted Data']])
    print(f'Result - Total: {length} links')  
    print(f'Result - Successful: {len(ID)} links, Failed: {len(problem_ID)} links')

    cleaned_df = pd.DataFrame({'ID': ID, 'Link': Link,'Title': Title, 'Text': Text})
    return cleaned_df

def filter_links(cleaned_df, best_doc):
    # Filter relevant links based on document similarity with best_doc
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    embed = hub.load(module_url) # if ValueError > delete the folder and rerun 

    def similarity_score(doc1, doc2):
        x = embed([doc1, doc2])
        corr = np.inner(x, x)
        return corr[0, 1]

    scores = []
    for index, row in cleaned_df.iterrows():
        header = row.Title
        score = similarity_score(best_doc, header)
        scores.append(score)

    cleaned_df['Score'] = scores
    cleaned_df = cleaned_df.sort_values(by='Score', ascending=False)

    filtered_df = cleaned_df[cleaned_df.Score >= 0.25] # set threshold
    print(f"Result - Filtered: {len(filtered_df.Link)} links")
    return filtered_df

def extract_names(filtered_df):
    # Extract all names using NER
    def combine_similar_names(names):
        combined_names = defaultdict(int)

        for name in names:
            found = False
            for combined_name in combined_names:
                if name in combined_name or combined_name in name:
                    combined_names[combined_name] += 1
                    found = True
                    break
            if not found:
                combined_names[name] += 1

        return list(combined_names.keys())

    NER = spacy.load('en_core_web_sm')

    all_names = []
    for index, row in filtered_df.iterrows():
        article_text = row.Text
        article_ner = NER(article_text)
        text, label = [], []
        for word in article_ner.ents:
            text.append(word.text)
            label.append(word.label_)
        names_df = pd.DataFrame(list(zip(text, label)), columns=['name', 'label'])
        raw_names = list(names_df[names_df.label == 'PERSON'].name)
        combined_names = combine_similar_names(raw_names)
        all_names.append(combined_names)

    filtered_df['All Names'] = all_names

    return filtered_df

def identify_criminals(filtered_df):
    model_name = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    
    q1 = "Who is the criminal?"
    q2 = "Who is the suspect?"
    context_list = filtered_df['Text']
    links = filtered_df['Link']
    titles = filtered_df['Title']
    q1_answers  = []
    q2_answers = []
    i=0
    for context in context_list:
        response1 = nlp({
            'question': q1,
            'context': context
        })
        q1_answers.append(response1)
        response2 = nlp({
            'question': q2,
            'context': context
        })
        q2_answers.append(response2)
        i+=1
        print(i)
    llm_df = pd.DataFrame({
        'Links': links,
        'Titles': titles,
        'Text':context_list,
        'Q1 Answers': q1_answers,
        'Q2 Answers': q2_answers
    })

    llm_answer_q1 = [row['answer'] for row in llm_df['Q1 Answers']]
    llm_answer_q2 = [row['answer'] for row in llm_df['Q2 Answers']]
    
    # clean llm answers: handle multiple names
    def split_string(text):
        if "and " in text:
            return text.split(" and ")
        else:
            return [text]
    
    llm_df['Q1 Name'] = list(map(split_string, llm_answer_q1))
    llm_df['Q2 Name'] = list(map(split_string, llm_answer_q2))
    llm_df['All Names'] = filtered_df['All Names']
    return llm_df

def match_names(llm_df):
    
    # function for matching 2 strings
    def find_contained_string(string1, string2):
        for i in range(len(string2)):
            for j in range(i + 1, len(string2) + 1):
                substring = string2[i:j]
                if string1 in substring:
                    return substring
        return None
    criminals = []
    index_matched = []
    for index, row in llm_df.iterrows():
        
        criminal_q1 = []
        for name in row['All Names']:
            for llm_name in row['Q1 Name']:
                match = find_contained_string(llm_name, name)
                if match is not None:
                    criminal_q1.append(match)
        
        if len(criminal_q1) > 0:
            criminals.append(criminal_q1)
            index_matched.append(index)
        else:
            criminal_q2 = []
            for name in row['All Names']:
                for llm_name in row['Q2 Name']:
                    match = find_contained_string(llm_name, name)
                    if match is not None:
                        criminal_q2.append(match)
            
            if len(criminal_q2) > 0:
                criminals.append(criminal_q2)
                index_matched.append(index)
    
    criminals = [name[0] for name in criminals]
    criminal_df = llm_df.loc[index_matched]
    criminal_df["Criminal"] = criminals
    criminal_df = criminal_df[['Links', 'Titles', 'Criminal']]
    criminal_df = criminal_df.drop_duplicates(subset=['Criminal'])

    print(f"Result - Matched: {len(criminal_df.Links)} links")
    return criminal_df

def test_results(criminal_df):
    kyc_df = pd.read_csv('../results/all_names_kyc.csv')
    extracted_names = criminal_df.Criminal.to_list()
    extracted_names = list(set([word.upper() for word in extracted_names])) # remove duplicates, upper case
    kyc_names = [remove_prefix(name.upper()) for name in kyc_df['name'].tolist()]
    common_names = [item for item in extracted_names if item in kyc_names]
    criminal_df['Criminal'] = criminal_df['Criminal'].str.upper()
    found_df = criminal_df[criminal_df['Criminal'].isin(common_names)].drop_duplicates(subset=['Criminal'])
    return found_df

def remove_prefix(name):
    prefixes = ['DR. ', 'DR.', 'MR. ', 'MR.', 'MRS. ', 'MRS.']
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix):].strip()
    return name

def main():
    start_time = time.time()

    scraped_data, best_doc = load_data()
    print("Cleaning data....")
    cleaned_df = clean_data(scraped_data)
    
    print("Filtering links....")
    filtered_df = filter_links(cleaned_df, best_doc)
    
    print("Extracting names....")
    filtered_df = extract_names(filtered_df)
    
    print("Identifying criminals....")
    llm_df = identify_criminals(filtered_df)
    llm_df.to_csv('../results/llm_df.csv')
    print("LLM Output Generated!!")

    print("Matching names....")
    criminal_df = match_names(llm_df)
    criminal_df.to_csv('../results/criminal_df.csv')
    print("Output File 1 Generated!!")
    
    print("Testing Result....")
    found_df = test_results(criminal_df)
    found_df.to_csv('../results/found_df.csv')
    print("Output File 2 Generated!!")
    
    print(f"Final Results - Found {len(found_df.Criminal)} names")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time} seconds")

if __name__ == "__main__":
    main()