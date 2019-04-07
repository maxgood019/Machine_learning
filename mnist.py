# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 18:14:01 2019

@author: user
"""

import tensorflow as tf

tf.__version__ #1.13.1

mnist = tf.keras.datasets.mnist #28X28 images of hand-written digits

(x_train,y_train), (x_test,y_test) = mnist.load_data()

#scale data
x_train = tf.keras.utils.normalize(x_train,axis = 1 )
x_test = tf.keras.utils.normalize(x_test,axis = 1 )

model  = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(x_train,y_train,epochs=10)

val_loss, val_acc = model.evaluate(x_test,y_test)

model.save('num_reader.model')

new_model = tf.keras.models.load_model('num_reader.model')

predictions = new_model.predict(x_test)

import numpy as np
print(np.argmax(predictions[1]))

plt.imshow(x_test[1])
plt.show()



#import matplotlib.pyplot as plt

#print(x_train[0])
#plt.imshow(x_train[0])
#plt.show()