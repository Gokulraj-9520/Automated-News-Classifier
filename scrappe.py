import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import nltk
import warnings 
warnings.filterwarnings("ignore")
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scrape_sport import get_sports_news
from scrape_business import get_business_news
from scrape_weather import get_weather_news
from scrape_politics import get_politics_news
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC
import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

pd.options.display.max_columns=None
pd.options.display.max_rows=None
pd.options.display.max_colwidth=None
lemmatizer=WordNetLemmatizer()

df_sports=get_sports_news()
df_business=get_business_news()
df_weather=get_weather_news()
df_politics=get_politics_news()
dfs = [df_sports, df_business, df_weather, df_politics]

# concatenate the dataframes vertically
df = pd.concat(dfs, axis=0, ignore_index=True)

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['content'])

# Applying K-means clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(tfidf_matrix)

# Getting the cluster labels for each article
df['cluster'] = kmeans.labels_

oe=OrdinalEncoder()
df['topic']=oe.fit_transform(df[['topic']])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['content'], df['topic'], test_size=0.25, random_state=42)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Testing the model's performance
X_test_vectorized = vectorizer.transform(X_test)

knn=KNeighborsClassifier()
knn.fit(X_train_vectorized,y_train)
knn_preds=knn.predict(X_test_vectorized)
print(accuracy_score(knn_preds,y_test))

print(classification_report(knn_preds,y_test))

with open('model.pkl', 'wb') as file:
    pickle.dump(knn, file)

with open('vector.pkl','wb') as file:
    pickle.dump(vectorizer,file)

with open('model.pkl','rb') as file:
    loaded_model=pickle.load(file)

with open('vector.pkl','rb') as file:
    loaded_vector=pickle.load(file)

raw_data=['The French beauty empire is on track for its best stock market performance in decades.']

vector_data=loaded_vector.transform(raw_data)

print(loaded_model.predict(vector_data))


