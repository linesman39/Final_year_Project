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
y=df1["Category"]
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
        if prediction1=="1":
            st.warning("Spam Text!!")
        elif prediction1=="0":
            st.success("Ham Text!!")

#Bernoulli spam detection Prediction 

model2=BernoulliNB()
model2.fit(x_train1,y_train1)

#Bernoulli spam detection Analysis Page
if rad=="Bernoulli Naive Bayes Detection":
    st.header("Detect Whether A Text Is Spam Or Ham??")
    sent2=st.text_area("Enter The Text")
    transformed_sent2=transform_text(sent2)
    vector_sent=tfidf1.transform([transformed_sent2])
    prediction=model2.predict(vector_sent)[0]

    if st.button("Predict"):
        if prediction=="1":
            st.warning("Spam Text!!")
        elif prediction=="0":
            st.success("Ham Text!!")
#confusion matrix
#----

model2=BernoulliNB()
model2.fit(x_train1,y_train1)

#Confusion matrix
if rad=="confusion matrix":
    st.header("Confusion matrix of multinomial and bernoulli naive bayes")
    st.image("Spam and ham detection.jpeg")
    if rad=="Bernoulli Naive Bayes Detection":
    st.header("Detect Whether A Text Is Spam Or Ham??")
    sent2=st.text_area("Enter The Text")
    transformed_sent2=transform_text(sent2)
    vector_sent=tfidf1.transform([transformed_sent2])
    prediction=model2.predict(vector_sent)[0]

    if st.button("Predict"):
        if prediction=="1":
            st.warning("Spam Text!!")
        elif prediction=="0":
            st.success("Ham Text!!")
#Other info
#----

model2=BernoulliNB()
model2.fit(x_train1,y_train1)

#Other info
if rad=="Other info":
    st.header("Analysis of multinomial and bernoulli naive bayes")
    st.image("Spam and ham detection.jpeg")
if rad=="Bernoulli Naive Bayes Detection":
    st.header("Detect Whether A Text Is Spam Or Ham??")
    sent2=st.text_area("Enter The Text")
    transformed_sent2=transform_text(sent2)
    vector_sent=tfidf1.transform([transformed_sent2])
    prediction=model2.predict(vector_sent)[0]

    if st.button("Predict"):
        if prediction=="1":
            st.warning("Spam Text!!")
        elif prediction=="0":
            st.success("Ham Text!!")
#Conclusion
#----
model2=BernoulliNB()
model2.fit(x_train1,y_train1)

#Conclusion
if rad=="Conclusion":
    st.header("Conclusiion from comparative analysis of multinomial and bernoulli naive bayes for SMS spam detection")
    st.image("Spam and ham detection.jpeg")
    sent2=st.text_area("Enter The Text")
    transformed_sent2=transform_text(sent2)
    vector_sent=tfidf1.transform([transformed_sent2])
    prediction=model2.predict(vector_sent)[0]

    if st.button("Predict"):
        if prediction=="1":
            st.warning("Spam Text!!")
        elif prediction=="0":
            st.success("Ham Text!!")
