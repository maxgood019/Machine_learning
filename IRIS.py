# -*- coding: utf-8 -*-
"""
Created on Sun May  5 15:49:19 2019

@author: user
"""

import pandas as pd


df = pd.read_csv('iris.csv')

df.columns = ['sepal_length','sepal_width','petal_length','petal_width','target']

df['target'] = df['target'].apply(int)

y = df['target']
X = df.drop('target',axis = 1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3)

import tensorflow as tf
#feature columns
feat_cols  = [] #list
for col in X.columns:
    feat_cols.append(tf.feature_column.numeric_column(col))

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train
                                   ,y=y_train,
                                   batch_size=10,
                                   num_epochs=5,
                                   shuffle = True)
classifier = tf.estimator.DNNClassifier(
        hidden_units = [10,20,10],n_classes = 3, 
        feature_columns = feat_cols)
classifier.train(input_fn = input_func,steps=50)

#evaluate the performance
#no need to input y
pred_fn = tf.estimator.inputs.pandas_input_fn(
        x=X_test, batch_size = len(X_test),shuffle=False)

predictions = list(classifier.predict(input_fn = pred_fn))

final_preds = []
for pred in predictions:
    final_preds.append(pred['class_ids'][0])

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,final_preds))
print(classification_report(y_test,final_preds))


 
            
