import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

data = pd.read_csv('modified_churn_data.csv')
user_id = data['user']
data = data.drop(columns = ['user'])

#one hot encoding
data.housing.value_counts()
data = pd.get_dummies(data) #categorical columns will be separated
data = data.drop(columns = ['housing_na','zodiac_sign_na','payment_type_na'])

#splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                        data.drop(columns = 'churn'),
                                        data['churn'],
                                        test_size = 0.2, random_state = 0) #20 for test set, 80 for training set
#balance the training set
y_train.value_counts()             
pos_index = y_train[y_train == 1 ].index
neg_index = y_train[y_train == 0 ].index                                

if len(pos_index) > len(neg_index):
    higher = pos_index
    lower = neg_index 
else:
    lower = pos_index
    higher = neg_index
random.seed(0)
higher = np.random.choice(higher, size=len(lower))
lower = np.asarray(lower)
new_indexes = np.concatenate((lower, higher))

X_train = X_train.loc[new_indexes,]
y_train = y_train[new_indexes]
# feature scaling #normalize the numerical data
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
#scaler return np array, we will lose index and column name, thus transfer to dataframe
X_train2  = pd.DataFrame(sc_x.fit_transform(X_train)) 
X_test2   = pd.DataFrame(sc_x.transform(X_test))

X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values

X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2

#Build Model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0 )
classifier.fit(X_train,y_train)

#pred 
y_pre = classifier.predict(X_test)

#evaluate
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score,f1_score, precision_score
cm = confusion_matrix(y_test, y_pre)
ac = accuracy_score(y_test, y_pre)
re = recall_score(y_test, y_pre)
pre = precision_score(y_test, y_pre)
f1 = f1_score(y_test, y_pre)

results = pd.DataFrame(columns=['Model Name','AC','Re','PRE','F1'],
                       data = [['Logistic',ac,re,pre,f1]])

#confusion matrix figure 
df_cm = pd.DataFrame(cm, index = (0,1), columns = (0,1))
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot = True, fmt='g')
print('Data Accuracy: %0.4f' %ac)

#K fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator= classifier,
                             X= X_train,
                             y= y_train,
                             cv = 10)
accuracies.mean()

#analyzing the coefficients


pd.concat([pd.DataFrame(X_train.columns, columns = ['features']),
                         pd.DataFrame(np.transpose(classifier.coef_), columns=['coef'])], #np.transpose(rows to coulmns)
                        axis = 1 )#axis =1 #columns not rows


####feature  selection 

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#model test
classifier = LogisticRegression()
rfe = RFE(classifier, 20)

rfe = rfe.fit(X_train,y_train)

#check the selection of the attributes
print(rfe.support_)
X_train.columns[rfe.support_]
rfe.ranking_

#Again
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train[X_train.columns[rfe.support_]],y_train)

#predict
y_pre = classifier.predict(X_test[X_test.columns[rfe.support_]])

cm = confusion_matrix(y_test, y_pre)
ac = accuracy_score(y_test, y_pre)
re = recall_score(y_test, y_pre)
pre = precision_score(y_test, y_pre)
f1 = f1_score(y_test, y_pre)

models_feature_selection = pd.DataFrame(columns=['Model Name','AC','Re','PRE','F1'],
                       data = [['Logistic_feature_selection',ac,re,pre,f1]])
results = results.append(models_feature_selection,ignore_index = True)

df_cm = pd.DataFrame(cm, index = (0,1), columns = (0,1))
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot = True, fmt='g')

#analyzing the coefficients


pd.concat([pd.DataFrame(X_train.columns[rfe.support_], columns = ['features']),
                         pd.DataFrame(np.transpose(classifier.coef_), columns=['coef'])], #np.transpose(rows to coulmns)
                        axis = 1 )#axis =1 #columns not rows

final_report = pd.concat([y_test, user_id], axis = 1 ).dropna()
final_report['predict_churn']  = y_pre
final_report = final_report[['user','churn','predict_churn']].reset_index(drop=True)














