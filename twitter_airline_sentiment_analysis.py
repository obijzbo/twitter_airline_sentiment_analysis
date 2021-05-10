#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 16:26:50 2021

@author: tns
"""

import pandas as pd
import nltk
tweets=pd.read_csv('Tweets.csv')
print(tweets.head())
print(tweets.shape)

#Data processing
tweets_df=tweets.drop(tweets[tweets['airline_sentiment_confidence']<0.5].index,axis=0)
print(tweets_df.shape)
x=tweets_df['text']
y=tweets_df['airline_sentiment']

#Cleaning the data

from nltk.corpus import stopwords
nltk.download('stopwords')
import string
from nltk.stem import PorterStemmer

stop_words=stopwords.words('english')
punct=string.punctuation
stemmer=PorterStemmer()

import re
cleaned_data=[]
for i in range(len(x)):
  tweet=re.sub('[^a-zA-Z]',' ',x.iloc[i])
  tweet=tweet.lower().split()
  tweet=[stemmer.stem(word) for word in tweet if (word not in stop_words) and (word not in punct)]
  tweet=' '.join(tweet)
  cleaned_data.append(tweet)
  
print(cleaned_data)
print(x)
print(y)

sentiment_ordering = ['negative', 'neutral', 'positive']

y = y.apply(lambda x: sentiment_ordering.index(x))

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=3000,stop_words=['virginamerica','unit'])
x_fit=cv.fit_transform(cleaned_data).toarray()
x_fit.shape

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
model=MultinomialNB()

X_train,X_test,y_train,y_test=train_test_split(x_fit,y,test_size=0.3)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn.metrics import classification_report
cf=classification_report(y_test,y_pred)
print(cf)

text_file = open("Output.txt", "w")
text_file.write(cf)
text_file.close()