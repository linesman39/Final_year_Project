<h1 align="center">
             Comparative analysis of multinomial and bernoulli naive bayes for SMS spam detection Web App 💬 📝 ✍️
</h1>

![image](https://user-images.githubusercontent.com/78029145/154792740-dadca757-5424-4e4c-af69-fc3a5055af3b.png)

This app is used to perform an indepth analysis of the performance of multinomial and bernoulli naive bayes for SMS spam detection
The analysis sections include ->

**1. Multinomial Naive bayes**

**2. Bernoulli Naive bayes**


## Tech Stacks Used

<img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/>

## Libraries Used

<img src="https://img.shields.io/badge/numpy%20-%2314354C.svg?&style=for-the-badge&logo=numpy&logoColor=white"/> <img src="https://img.shields.io/badge/pandas%20-%2314354C.svg?&style=for-the-badge&logo=pandas&logoColor=white"/> <img src="https://img.shields.io/badge/streamlit%20-%2314354C.svg?&style=for-the-badge&logo=streamlit&logoColor=white"/> <img src="https://img.shields.io/badge/nltk%20-%2314354C.svg?&style=for-the-badge&logo=nltk&logoColor=white"/> <img src="https://img.shields.io/badge/scikitlearn%20-%2314354C.svg?&style=for-the-badge&logo=scikitlearn&logoColor=white"/>

## Structure Of The Project

- Each prediction page is conneceted with a Machine Learning Model which uses either of multinomial or bernoulli naive bayes to predict the results.
- Also we have thesame dataset being used for each prediction.
- We can land into each prediction site of the web app from the options in the Navigation Menu.
- We have only 1 relevant feature taken into consideration which is the text and then the text is preprocessed and vectoized with help of TF-IDF Vectorizer to fit into the model and tain it.
- So the user gets a broad overview of the text after the analysis

## The feature taken into consideration

| Text Analysis Type | Feature |
| - | - |
| multinomial naive bayes Page | Text |
| bernoulli naive bayes Page | Text |


The text is preprocessed then fed to the model.

## Deployment Of The Project

After the modeling part the model is deployed using Streamlit library on Streamlit Share so that the app is available for usage for everyone.

## Link To My Web Application -

https://share.streamlit.io/bhaswatiroy/complete-text-analysis-streamlit-web-app/main/app.py

## Glance At The Hosted Application- 

### 1. Home Page
![image](https://user-images.githubusercontent.com/78029145/154792997-c60376bb-411a-4624-aeeb-f552416a8cfb.png)

### 2. Spam or Ham Detection Page
![image](https://user-images.githubusercontent.com/78029145/154802534-75818785-70a8-46ff-99cc-adfef7b0c95b.png)


