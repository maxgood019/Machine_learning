
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import seaborn as sns

random.seed(100)

#data preprocessing
dataset = pd.read_csv('Financial-Data.csv')

dataset = dataset.drop(columns = ['months_employed']) #unreasonable data column

dataset['personal_account_months'] = (dataset.personal_account_m + (dataset.personal_account_y)*12)
#dataset[['personal_account_months', 'personal_account_y', 'personal_account_m']].head()
dataset = dataset.drop(columns = ['personal_account_y','personal_account_m'])

#one hot encoding
dataset = pd.get_dummies(dataset)
dataset = dataset.drop(columns=['pay_schedule_semi-monthly'])

#remove extra data
e_signed = dataset["e_signed"]
user_id = dataset['entry_id']

dataset = dataset.drop(columns = ['e_signed','entry_id'])

# train_test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
                                    dataset, e_signed, test_size=0.2, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train2 = pd.DataFrame(sc.fit_transform(X_train))
X_test2 = pd.DataFrame(sc.transform(X_test))

X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values

X_train2.index = X_train.index.values
X_test2.index = X_test.index.values  


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score

#---model LogisticRegression---
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, penalty='l1')
classifier.fit(X_train,y_train)

#predict
y_pred_LR = classifier.predict(X_test)

#report


acc = accuracy_score(y_test, y_pred_LR)
prec = precision_score(y_test, y_pred_LR)
rec = recall_score(y_test, y_pred_LR)
f1 = f1_score(y_test, y_pred_LR)

model_result = pd.DataFrame([['Linear Regression', acc, prec, rec, f1]],
             columns = ['Model','Accuracy','Precision','Recall','F1_Score']
             )
#---End---

#---model SVM(linear)---
from sklearn.svm import SVC
classifier = SVC(random_state = 0, kernel='linear')
classifier.fit(X_train,y_train)

#predict 
y_pred_SVM = classifier.predict(X_test)

acc = accuracy_score(y_test,y_pred_SVM)
recall = recall_score(y_test,y_pred_SVM)
f1 = f1_score(y_test,y_pred_SVM)
prec = precision_score(y_test,y_pred_SVM)

model_SVM = pd.DataFrame([['SVM_Linear',acc,prec,rec,f1]],
             columns = ['Model','Accuracy','Precision','Recall','F1_Score'])

model_result = model_result.append(model_SVM, ignore_index= True)

#---End---

#---SVM_RBF---
classifier = SVC(random_state = 0, kernel='rbf')
classifier.fit(X_train,y_train)

y_pred_SVMRBF = classifier.predict(X_test)

acc = accuracy_score(y_test,y_pred_SVMRBF)
recall = recall_score(y_test, y_pred_SVMRBF)
f1 = f1_score(y_test,y_pred_SVMRBF)
pre = precision_score(y_test,y_pred_SVMRBF)

model_SVM_RBF = pd.DataFrame(columns = ['Model','Accuracy','Precision','Recall','F1_Score'],
                                data = [['SVM_RBF',acc,prec,recall,f1]])

model_result = model_result.append(model_SVM_RBF,ignore_index = True)
#---End---

#---random forest---

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state= 0 , n_estimators= 100,
                                    criterion= 'entropy')
classifier.fit(X_train,y_train)

y_pred_RF = classifier.predict(X_test)

acc = accuracy_score(y_test,y_pred_RF)
recall = recall_score(y_test,y_pred_RF)
f1 = f1_score(y_test,y_pred_RF)
pre = precision_score(y_test, y_pred_RF)

model_result_RF = pd.DataFrame(columns = ['Model','Accuracy','Precision','Recall','F1_Score'],
                                data = [['RF',acc,prec,recall,f1]])
#model_result_RF  = pd.DataFrame(columns=['Model','Accuracy','Precision','Recall','F1_Score'],
#                                data =  [['RF',acc,recall,pre,f1]])
model_result = model_result.append(model_result_RF,ignore_index = True)

#---End---


#---K-fold cross validation---
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator=classifier,X = X_train, y = y_train,
                           cv = 10) #Random Forest classifier
print("Random Forest Classification Accuracy: %0.2f (+/- %0.2f)" %(accuracy.mean(),accuracy.std()*2))


#---End---

#---Grid Search---
#fine tune the model of Random Forest
#Entropy
para = {"max_depth":[3,None],
        "max_features":[1,5,10],
        "min_samples_split":[2,5,10],
        "min_samples_leaf": [1,5,10],
        "bootstrap":[True,False],
        "criterion":["entropy"]}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier,#model that want to tune
                           param_grid = para,
                           scoring = "accuracy",
                           cv = 10,
                           n_jobs = -1)
#estimate the time
t0 = time.time()
grid_search = grid_search.fit(X_train,y_train)
t1 = time.time()
print("total time %0.2f secs" %(t1-t0))


rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters
#---End---

#Entropy 2
parameters = {"max_depth": [None],
              "max_features": [3, 5, 7],
              'min_samples_split': [8, 10, 12],
              'min_samples_leaf': [1, 2, 3],
              "bootstrap": [True],
              "criterion": ["entropy"]}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier, # Make sure classifier points to the RF model
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10,
                           n_jobs = -1)

t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters
#End

#Predict test set
y_pred = grid_search.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results = pd.DataFrame([['Random Forest (After fine tune entropy)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

model_result = model_result.append(results, ignore_index = True)

#end
# Round 1: Gini
parameters = {"max_depth": [3, None],
              "max_features": [1, 5, 10],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10],
              "bootstrap": [True, False],
              "criterion": ["gini"]}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier, # Make sure classifier points to the RF model
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10,
                           n_jobs = -1)

t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters

# Round 2: Gini
parameters = {"max_depth": [None],
              "max_features": [8, 10, 12],
              'min_samples_split': [2, 3, 4],
              'min_samples_leaf': [8, 10, 12],
              "bootstrap": [True],
              "criterion": ["gini"]}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier, # Make sure classifier points to the RF model
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10,
                           n_jobs = -1)

t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters


# Predicting Test Set
y_pred = grid_search.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Random Forest (n=100, GSx2 + Gini)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)


## Confusion Matrix
cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))




final_results = pd.concat([y_test, user_id], axis = 1).dropna()
final_results['predictions'] = y_pred
final_results = final_results[['entry_id', 'e_signed', 'predictions']]








