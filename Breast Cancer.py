import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

df_cancer = pd.DataFrame(np.c_[cancer['data'],cancer['target']], columns = np.append(cancer['feature_names'],['target']))

sns.pairplot(df_cancer, hue = 'target', vars=['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )

sns.countplot(df_cancer['target'])


plt.figure(figsize=(20,10))
sns.heatmap(df_cancer.corr(),annot=True)



#model training 
X = df_cancer.drop(['target'],axis = 1)
y = df_cancer['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(  X, y, test_size=0.2, random_state=5)

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
svc_model = SVC()

#normalize
min_train = X_train.min()
range_train = (X_train - min_train).max()
X_train_scaled = (X_train - min_train) / range_train

min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test) / range_test


sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)

svc_model.fit(X_train_scaled,y_train)

y_prediction = svc_model.predict(X_test_scaled)
cm1 = confusion_matrix(y_test, y_prediction)
sns.heatmap(cm1, annot=True)
print(classification_report(y_test, y_prediction))
#tune SVM
param_grid = {'C':[0.1,1,10,100], 'gamma':[1,0.1,0.01,0.001], 'kernel':['rbf']}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 4 )
grid.fit(X_train_scaled, y_train)
# grid.best_params_
#{'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}
grid_pred = grid.predict(X_test_scaled)
cm = confusion_matrix(y_test, grid_pred)
sns.heatmap(cm,annot=True)
print(classification_report(y_test,grid_pred))


