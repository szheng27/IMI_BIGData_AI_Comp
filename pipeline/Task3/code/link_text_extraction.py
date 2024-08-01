#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3 Created in Feb 2023

Team X
2023-2024 IMI Big Data and Artificial Intelligence Competition
@author: Tushar Raju, Amelyn Wang
"""
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import csv

def news_get_text(links,scraped_data):
    text_list = []
    for link in links:
        response = requests.get(link)
        soup = BeautifulSoup(response.content, 'html.parser')
        p_tags = soup.find_all('p')
        text = '\n'.join(p.get_text() for p in p_tags)
        html_tags_pattern = re.compile(r'<.*?>')
        text_without_tags = re.sub(html_tags_pattern, '', text)
        if link.startswith("https://www.justice.gov/opa/pr/"):
            title = soup.find_all('h1',class_="page-title")
        elif link.startswith("https://www.cbc.ca/"):
            title = soup.find_all('h1',class_="detailHeadline")
        elif link.startswith("https://www.timminstoday.com/"):
            title = soup.find_all('h1',class_="title details-title")
        elif link.startswith("https://globalnews.ca/"):
            title = soup.find_all('h1',class_="l-article__title")
        elif link.startswith("https://www.saobserver.net"):
            title = soup.find_all('h1',class_="title details-title")
        elif link.startswith("https://www.newswire.ca/news-releases/"):
            title = soup.find_all('h1')
        elif "ctvnews" in link:
            title = soup.find_all('h1',class_="c-title__text")
        elif link[-7:-4].isdigit():
            title = soup.find_all('h1',class_="c-title__text")
        else:
            title_without_tags = ""
        text = '\n'.join(t.get_text() for t in title)
        title_without_tags = re.sub(html_tags_pattern, '', text)
        scraped_data[link] = {'Title':title_without_tags,'Text':text_without_tags}
        text_list.append(text_without_tags)
        print(len(scraped_data))
    return text_list

def export_links_and_titles_to_csv(dictionary, filename):
    with open(filename, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['Link', 'Extracted Data']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for link, title in dictionary.items():
            writer.writerow({'Link': link, 'Extracted Data': title})

def read_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

def main(): 
    all_links = pd.read_csv("../results/all_news_links.csv")
    scraped_data = {}
    text_data = news_get_text(all_links['News Links'],scraped_data)
    #text_data = news_get_text(["https://globalnews.ca/news/10294948/kansas-city-shooting-super-bowl-parade/","https://www.cbc.ca/news/canada/calgary/eagle-parts-trafficking-case-hears-sentencing-arguments-1.2534354","https://www.saobserver.net/news/1-arrested-for-trafficking-black-bear-parts-in-bc-5559567","https://www.justice.gov/opa/pr/attorney-general-merrick-b-garland-honors-justice-department-employees-and-partners-70th-and","https://www.timminstoday.com/national-news/trudeau-visits-us-shopify-shareholders-vote-on-founder-share-in-the-news-for-june-7-5451002","https://bc.ctvnews.ca/b-c-man-charged-for-trafficking-wildlife-meat-cos-1.6754907","https://www.newswire.ca/news-releases/operation-northern-fur-leads-to-20-000-in-fines-and-a-prohibition-order-for-a-manitoba-resident-who-illegally-imported-exported-and-transported-wildlife-881334620.html"],scraped_data)
    export_links_and_titles_to_csv(scraped_data, "../results/links_and_titles.csv")
    extracted_data = read_data("../results/links_and_titles.csv")

if __name__ == "__main__":
    main()
