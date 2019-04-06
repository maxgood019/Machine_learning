import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Restaurant_Reviews.tsv',
                      delimiter = '\t', #TAB
                      quoting = 3 #ignore the double quote
                      )

#dataset['Review'][0]
#clean the text, such as the, and, ..., loved (past tense) etc...
import re
import nltk
nltk.download('stopwords') #download the list of irrelevant words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
result = []
for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    # def remove_html_tags(text):
    #"""Remove html tags from a string"""
    #import re
    #clean = re.compile('<.*?>')
    #return re.sub(clean, '', text)
    #review = review.lower()

    review = review.split() # to list
    ps = PorterStemmer()
    #steming such as loved to love #set remove duplicated 
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    review = ' '.join(review)
    result.append(review)

#create bag of words model 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 1500 ) #reduce the number of most frequent word
#https://medium.com/@jorlugaqui/how-to-strip-html-tags-from-a-string-in-python-7cb81a2bbf44
X = cv.fit_transform(result).toarray()
y = dataset.iloc[: , 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2, random_state = 0)


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
acc = (cm[0][0]+cm[1][1]) / 200
(55+91) / 200 
