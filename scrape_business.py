# https://www.bbc.com/news/business
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
lemmatizer=WordNetLemmatizer()

def get_business_news():
    def get_bbc_news():
        url = "https://www.bbc.com/news/business"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = soup.find_all('div', class_='gs-c-promo-body gs-u-mt@xxs gs-u-mt@m gs-c-promo-body--flex gs-u-mt@xs gs-u-mt0@xs gs-u-mt--@s gs-u-mt--@m gel-1/2@xs gel-1/1@s')
            news_list=[]
            for news_item in news_items:
                news_dict={}
                try:
                    news_dict['title']=news_item.find('h3', class_='gs-c-promo-heading__title gel-pica-bold nw-o-link-split__text').text
                    news_dict['content']=news_item.find('p',class_='gs-c-promo-summary gel-long-primer gs-u-mt nw-c-promo-summary gs-u-display-none gs-u-display-block@l').text
                    news_list.append(news_dict)
                except AttributeError:
                    pass
                
            return news_list
    news = get_bbc_news()
    for item in news:
        print('title',item['title'])
        print('content',item['content'])

    df=pd.DataFrame()
    for item in news:
        print(item['title'])
        print(item['content'])
        row = pd.Series([item['title'], item['content'], 'Business'])
        df = df.append(row, ignore_index=True)

    df.columns = ['title', 'content', 'topic']

    def lowercase(text):
        return text.lower()

    def remove_non_alphabetic(text):
        return re.sub(r'\W+',' ',text)

    def tokenize_text(text):
        return word_tokenize(text)

    def remove_stopwords(word_list):
        stopwords_value=set(stopwords.words('english'))
        return [word for word in word_list if word not in stopwords_value]

    def lemmatize_words(word_list):
        return [lemmatizer.lemmatize(word) for word in word_list ]

    def clean_and_preprocess(text):
        text=lowercase(text)
        text=remove_non_alphabetic(text)
        text=tokenize_text(text)
        text=remove_stopwords(text)
        text=lemmatize_words(text)
        return text
    processed_titles=[]
    processed_contents=[]
    for item in news:
        processed_title=clean_and_preprocess(item['title'])
        processed_content=clean_and_preprocess(item['content'])
        processed_titles.append(processed_title)
        processed_contents.append(processed_content)
    df['processed_title'] = processed_titles
    df['processed_content'] = processed_contents
        
    return df
