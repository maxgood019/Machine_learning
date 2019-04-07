import numpy as np
import pandas as pd
import keras 
import matplotlib.pyplot as plt
import itertools  
from sklearn import svm, datasets 
from sklearn.metrics import confusion_matrix 
    #np.random.seed(2)


dataset = pd.read_csv('creditcard.csv')

from sklearn.preprocessing import StandardScaler
dataset['normalizedAmount'] = StandardScaler().fit_transform(dataset['Amount'].values.reshape(-1,1))

dataset = dataset.drop(['Amount'],axis = 1)
dataset = dataset.drop(['Time'],axis = 1 )

X = dataset.iloc[:,dataset.columns != 'Class']
y = dataset.iloc[:,dataset.columns == 'Class']

from sklearn.model_selection import train_test_split

X_train, x_test, y_train, y_test = train_test_split(X, y , test_size = 0.3, random_state = 0)

#X_train.shape

X_train = np.array(X_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential([
        Dense(units =16 ,input_dim = 29, activation = 'relu'),
        Dense(units =24, activation = 'relu'),
        Dropout(0.5),
        Dense(units =20, activation = 'relu'),
        Dense(units =24, activation = 'relu'),
        Dense(1, activation = 'sigmoid')
        ])
#model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_train,y_train,batch_size = 15, epochs=20)

score = model.evaluate(x_test,y_test)

#print(score)
#99.93914071369217%
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

y_pred = model.predict(x_test)
y_test = pd.DataFrame(y_test)

cnf_matrix = confusion_matrix(y_test,y_pred.round())
#array([[85273,    23],
       #[   29,   118]]