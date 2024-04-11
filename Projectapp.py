from tracemalloc import stop
import streamlit as st
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

nltk.download('punkt')
nltk.download('stopwords')
sw=nltk.corpus.stopwords.words("english")

rad=st.sidebar.radio("Navigation",["Home","Multinomial Naive Bayes Detection","Bernoulli Naive Bayes Detection","confusion matrix","Other Info","conclusion"])

#Home Page
if rad=="Home":
    st.title("Complete Spam or Ham Detection App")
    st.image("Spam and ham detection.jpeg")
    st.text(" ")
    st.text("The Following Options Are Available->")
    st.text(" ")
    st.text("1. Using Multinomial Naive Bayes Detection")
    st.text(" ")
    st.text("2. Using Bernoulli Naive Bayes Detection")
    st.text(" ")
    st.text("3. confusion matrix")
    st.text(" ")
    st.text("4. Other Info")
    st.text(" ")
    st.text("5. conclusion")


#function to clean and transform the user input which is in raw format
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    ps=PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

#Multinomial Spam Detection Prediction
tfidf1=TfidfVectorizer(stop_words=sw,max_features=20)
def transform1(txt1):
    txt1=tfidf1.fit_transform(txt1)
    return txt1.toarray()

df1=pd.read_csv("Spam Detection.csv")
x=transform1(df1["Text"])
y=df1["Label"]
x_train1,x_test1,y_train1,y_test1=train_test_split(x,y,test_size=0.1,random_state=0)
model1=MultinomialNB()
model1.fit(x_train1,y_train1)

#Multinomial Spam Detection Analysis Page
if rad=="Multinomial Naive Bayes Detection":
    st.header("Detect Whether A Text Is Spam Or Ham??")
    sent1=st.text_area("Enter The Text")
    transformed_sent1=transform_text(sent1)
    vector_sent1=tfidf1.transform([transformed_sent1])
    prediction1=model1.predict(vector_sent1)[0]

    if st.button("Predict"):
        if prediction1=="spam":
            st.warning("Spam Text!!")
        elif prediction1=="ham":
            st.success("Ham Text!!")

#Bernoulli spam detection Prediction 
tfidf2=TfidfVectorizer(stop_words=sw,max_features=20)
def transform2(txt2):
    txt2=tfidf2.fit_transform(txt2)
    return txt2.toarray()

df2=pd.read_csv("Spam Detection.csv")
x=transform1(df2["Text"])
y=df2["Label"]
x_train2,x_test2,y_train2,y_test2=train_test_split(x,y,test_size=0.1,random_state=0)
model2=BernoulliNB()
model2.fit(x_train2,y_train2)

#Bernoulli spam detection Analysis Page
if rad=="Bernoulli Naive Bayes Detection":
    st.header("Detect Whether A Text Is Spam Or Ham??")
    sent2=st.text_area("Enter The Text")
    transformed_sent2=transform_text(sent2)
    vector_sent2=tfidf2.transform([transformed_sent2])
    prediction2=model2.predict(vector_sent2)[0]

    if st.button("Predict"):
        if prediction2=="spam":
            st.warning("Spam Text!!")
        elif prediction2=="ham":
            st.success("Ham Text!!")
#confusion matrix
#----

#Confusion matrix
if rad=="confusion matrix":
    st.header("Confusion matrix of multinomial and bernoulli naive bayes")
    st.image("Spam and ham detection.jpeg")
#Other info
#----

#Other info
if rad=="Other info":
    st.header("Analysis of multinomial and bernoulli naive bayes")
    st.image("Spam and ham detection.jpeg")
#Conclusion
#----

#Conclusion
if rad=="Conclusion":
    st.header("Conclusiion from comparative analysis of multinomial and bernoulli naive bayes for SMS spam detection")
    st.image("Spam and ham detection.jpeg")
   

