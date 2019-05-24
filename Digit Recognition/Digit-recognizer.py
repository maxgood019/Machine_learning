import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import os

df_test = pd.read_csv('test.csv')
df_test_values = df_test.values

df_train = pd.read_csv('train.csv')
df_train_values = df_train.values

ans = pd.read_csv('sample_submission.csv')

x_train = df_train_values[:,1:]
y_train = df_train_values[:,0]


DT_C = RandomForestClassifier()
DT_C.fit(x_train,y_train)

y_pred = DT_C.predict(df_test_values)
y_pred = pd.Series(y_pred)
#draw:

fig,ax1 = plt.subplots(1)
ax1.imshow((df_test_values[5].reshape(28,28)),cmap='gray')
plt.show()

'''
test1 = pd.DataFrame([[23,5],[2,5],[15,78],[0,6]],
                     columns = ['hello','yo'])
test2 = pd.Series([11])

test3 = test1.join(test2.rename('yo1'),how= 'left')
''' 
ans = ans.drop(columns='Label',axis = 1 )
ans = ans.join(y_pred.rename('Label'))

ans.to_csv('submission_RF.csv',index= False)

'''
Using Keras CNN

'''

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPool2D
from keras.layers import Conv2D
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical
#x_train = x_train.reshape(42000,28,28,1)
X_train, X_test, y_train, y_test = train_test_split(
            x_train, y_train, test_size=0.2, random_state=0)

X_train = X_train.reshape(33600,28,28,1)
X_test = X_test.reshape(8400,28,28,1)
y_train1 = np.array(y_train)
y_test1 = np.array(y_test)

'''
# Transfer,for example as y_train(28140,) to (28140,10)
# [8] = [0,0,0,0,0,0,0,0,1,0]

'''
y_train1 = to_categorical(y_train)
y_test1 = to_categorical(y_test)

cnn  = Sequential()
cnn.add(Conv2D(32,kernel_size =(3,3),
               activation = 'relu',
               input_shape=(28,28,1)))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(32,(3,3),activation='relu'))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(32,(3,3),activation='relu'))
cnn.add(MaxPool2D(pool_size=(2,2)))#pooling layer

cnn.add(Flatten())
cnn.add(Dense(512,activation='relu'))
cnn.add(Dense(256,activation='relu')) #fully connected layer
cnn.add(Dropout(0.5))
cnn.add(Dense(units=10,activation='softmax')) #output layer

cnn.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

cnn.fit(X_train,y_train1,
        batch_size=128,
        epochs = 10,
        validation_data=(X_test,y_test1))


cnn.summary()
Accuracy = cnn.evaluate(X_test, y_test1,verbose = 0)
print("Accuracy_CNN: " ,Accuracy)

df_test_values_transform = df_test_values.reshape(-1,28,28,1)

#predict the test
results = cnn.predict(df_test_values_transform)
results = np.argmax(results, axis=1)
results = pd.Series(results)
ans = ans.drop(columns = ['Label'])
ans = ans.join(results.rename('Label'))

ans.to_csv('submission.csv',index = False)





