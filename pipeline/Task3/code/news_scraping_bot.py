#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3 Created in Feb 2023

Team X
2023-2024 IMI Big Data and Artificial Intelligence Competition
@author: Tushar Raju, Amelyn Wang
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time
import pandas as pd

# Function to scroll to the bottom of the page and click the "Load More" button until it disappears for Global News
def global_news_load_more_until_disappear(driver):
    while True:
        try:
            load_more_button = driver.find_element(by=By.ID,value="load-more-results")
            actions = ActionChains(driver)
            actions.move_to_element(load_more_button).perform()
            load_more_button.click()
            time.sleep(3)
        except:
            break

# Function to collect all the links on the Global News Page
def global_news_collect_links(driver):
    links = driver.find_elements(by=By.XPATH,value='//a[@href]')
    link_list = [link.get_attribute('href') for link in links]
    return link_list

# Function to scroll to the bottom of the page and click the "Load More" button until it disappears for CBC News
def cbc_news_load_more_until_disappear(driver):
    while True:
        try:
            load_more_button = driver.find_element(by=By.XPATH,value="//button[contains(text(),'Load More')]")
            driver.execute_script("arguments[0].scrollIntoView();", load_more_button)
            time.sleep(5)
            actions = ActionChains(driver)
            actions.move_to_element(load_more_button).perform()
            load_more_button.click()
            time.sleep(5)
        except:
            break

# Function to collect all the links on the CBC News page
def cbc_news_collect_links(driver):
    links = driver.find_elements(by=By.XPATH,value='//a[@href]')
    link_list = [link.get_attribute('href') for link in links]
    return link_list

# Function to collect links and move to next page for Salmon Arm Observer News
def load_next_page(driver):
    all_links = list()
    while True:
        try:
            links = driver.find_elements(by=By.XPATH,value='//a[@href]')
            link_list = [link.get_attribute('href') for link in links]
            all_links.extend(link_list)
            time.sleep(3)
            load_more_button = driver.find_element(by=By.XPATH,value='//*[@title="Next"]')
            actions = ActionChains(driver)
            actions.move_to_element(load_more_button).perform()
            load_more_button.click()
        except:
            break
    return all_links

# Function to collect links and move to next page for US Justice Government News
def load_next_page_jg(driver):
    all_links = list()
    while True:
        try:
            links = driver.find_elements(by=By.XPATH,value='//a[@href]')
            link_list = [link.get_attribute('href') for link in links]
            all_links.extend(link_list)
            time.sleep(3)
            load_more_button = driver.find_element(by=By.XPATH,value="//span[text()=' Next ']")
            actions = ActionChains(driver)
            actions.move_to_element(load_more_button).perform()
            load_more_button.click()
        except:
            break
    return all_links

# Function to collect links and move to next page for Timmins Today News
def load_next_page_tims(driver):
    all_links = list()
    while True:
        try:
            links = driver.find_elements(by=By.XPATH,value='//a[@href]')
            link_list = [link.get_attribute('href') for link in links]
            all_links.extend(link_list)
            time.sleep(3)
            load_more_button = driver.find_element(by=By.XPATH,value='//*[@title="Next"]')
            actions = ActionChains(driver)
            actions.move_to_element(load_more_button).perform()
            time.sleep(3)
            load_more_button.click()
        except:
            break
    return all_links

# Function to collect links and move to next page for Newswire
def collect_links_newswire(driver):
    all_links = list()
    try:
        links = driver.find_elements(by=By.XPATH,value='//a[@href]')
        link_list = [link.get_attribute('href') for link in links]
        all_links.extend(link_list)
        time.sleep(3)
    except:
        pass
    return all_links

# Function to collect links and move to next page for Salmon Arm Observer News
def collect_links_ctv(driver):
    all_links = list()
    while True:
        try:
            links = driver.find_elements(by=By.XPATH,value='//a[@href]')
            link_list = [link.get_attribute('href') for link in links]
            all_links.extend(link_list)
            time.sleep(3)
            load_more_button = driver.find_element(by=By.XPATH,value="//*[contains(text(), 'Next')]")
            actions = ActionChains(driver)
            actions.move_to_element(load_more_button).perform()
            time.sleep(3)
            load_more_button.click()
        except:
            break
    return all_links

def main():
    # Intializing Driver
    driver = webdriver.Chrome()
    all_links = list()
    # Keyword Search List
    keywords = ['wildlife trafficking','wildlife crime','wildlife fined','wildlife arrested','wildlife guilty','wildlife poaching']
    # For site formatting the url and accessing the links and adding on to a big link list of all
    for word in keywords:
        #Global News
        words = word.lower().replace(' ', '+')
        driver.get("https://globalnews.ca/?s="+words)
        time.sleep(3)
        global_news_load_more_until_disappear(driver)
        links = global_news_collect_links(driver)
        filtered_links = [link for link in links if link.startswith("https://globalnews.ca/news/")]
        all_links.extend(filtered_links)
        #CBC News
        words = word.lower().replace(' ', '%20')
        driver.get("https://www.cbc.ca/search?q="+words+"&section=news&sortOrder=date&media=all")
        time.sleep(5)
        cbc_news_load_more_until_disappear(driver)
        links = cbc_news_collect_links(driver)
        filtered_links = [link for link in links if link.startswith("https://www.cbc.ca/news/")]
        all_links.extend(filtered_links)
        #SAO News
        words = word.lower().replace(' ', '+')
        driver.get("https://www.saobserver.net/search?q="+words+"&Order=date&AuthorName=&BylineId=&DateStart=2012-01-01&DateEnd=")
        time.sleep(3)
        links = load_next_page(driver)
        filtered_links = [link for link in links if link.startswith("https://www.saobserver.net/news/")]
        all_links.extend(filtered_links)
        #JG News
        words = word.lower().replace(' ', '+')
        driver.get("https://www.justice.gov/news/press-releases?search_api_fulltext=+"+words+"&start_date=01%2F01%2F2012&end_date=&sort_by=field_date")
        time.sleep(5)
        links = load_next_page_jg(driver)
        filtered_links = [link for link in links if link.startswith("https://www.justice.gov/opa/pr/")]
        all_links.extend(filtered_links)
        #Timmin News
        words = word.lower().replace(' ', '+')
        driver.get("https://www.timminstoday.com/search?q="+words+"&Order=relevance&AuthorName=&BylineId=&DateStart=2012-01-01&DateEnd=")
        time.sleep(3)
        links = load_next_page_tims(driver)
        filtered_links = [link for link in links if link.startswith(("https://www.timminstoday.com/national-news/", "https://www.timminstoday.com/world-news/", "https://www.timminstoday.com/local-news/","https://www.timminstoday.com/police-beat","https://www.timminstoday.com/business-news/"))]
        all_links.extend(filtered_links)
        #Newswire
        words = word.lower().replace(' ', '%20')
        driver.get("https://www.newswire.ca/search/news/?keyword="+words)
        time.sleep(3)
        links = collect_links_newswire(driver)
        filtered_links = [link for link in links if link.startswith("https://www.newswire.ca/news-releases/") and link.endswith(".html")]
        all_links.extend(filtered_links)
        #CTV News
        words = word.lower().replace(' ', '+')
        driver.get("https://www.ctvnews.ca/search-results/search-ctv-news-7.137?q="+words)
        time.sleep(3)
        links = collect_links_ctv(driver)
        filtered_links = [link for link in links if link[-7:-4].isdigit()]
        all_links.extend(filtered_links)

    df = pd.DataFrame({'News Links':all_links})
    df = df.drop_duplicates()
    df.to_csv('../results/all_news_links.csv', index=False)
    driver.quit()
    print("Output File Generated")

if __name__ == "__main__":
    main()
