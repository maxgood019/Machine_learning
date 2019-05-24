import keras
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import PIL.Image as Image
from IPython.display import display



from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator    

pwd = "../Stanford Car Dataset by classes folder"
print(os.listdir(pwd))
train_dir = pwd+"/car_data/train"
test_dir  = pwd+"/car_data/test"
os.listdir(train_dir+"/")
#save all the brand and pic inside dic
train_brand_car = {}
for i in os.listdir(train_dir):
    train_brand_car[i] = os.listdir(train_dir + '/' + i)

#end save

#Get Car Brand
car_classes=[]
for i in train_brand_car:
    car_classes.append(i)
#End of Car Brand
    
#test dic
test={'hello':2,'yo':3}
test.values()
#test dic

car_img_list=[]
car_name_list=[]
for i,j in enumerate(train_brand_car.values()):
    for pic in j:
        car_img_list.append(pic)
        car_name_list.append(car_classes[i])
car_dir=[]
for i in range(len(car_name_list)):
    car_dir.append(train_dir + "/" + car_name_list[i] + "/" + car_img_list[i])
#verify:
plt.imshow(Image.open(car_dir[3]))
plt.title(car_name_list[3])
#end of verify
    
#pre view
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(15,15))

image1 = plt.imread(pwd+"/car_data/train/Audi S4 Sedan 2007/00159.jpg")
image2 = plt.imread(pwd+"/car_data/train/Aston Martin V8 Vantage Convertible 2012/00065.jpg")
image3 = plt.imread(pwd+"/car_data/train/Bentley Continental Flying Spur Sedan 2007/00057.jpg")
image4 = plt.imread(pwd+"/car_data/train/Bugatti Veyron 16.4 Coupe 2009/01249.jpg")

ax = axes[0,0] 
ax1 = axes[0,1]
ax2 = axes[1,0]
ax3 = axes[1,1]

ax.imshow(image1)
ax1.imshow(image2)
ax2.imshow(image3)
ax3.imshow(image4)
#end preview

#CSV summary
df = pd.DataFrame(data=[car_dir,car_name_list],index=['Dir','Brand']).T #.T reverse
df.to_csv('car_dir_brand.csv',index = False)
#end csv

#CNN para:
pic_width, pic_height = 256,256
nb_train_samples = len(car_name_list)
nb_validation_samples = 8041
epochs = 10 
steps_per_epoch = 256
batch_size = 64
n_classes = len(train_brand_car.values())
#end CNN para


train_datagen = ImageDataGenerator(rescale = 1./255,shear_range=0.2,
                                zoom_range=0.2,horizontal_flip=True,
                                rotation_range = 8)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_dir,
                                                target_size=(pic_width,pic_height),
                                                batch_size=batch_size,
                                                class_mode='categorical')
test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=(pic_width,pic_height),
                                             batch_size=batch_size,
                                             class_mode='categorical')



###CNN MODEL:
cnn = Sequential()

#first layer:
cnn.add(Conv2D(16,kernel_size=(5,5),input_shape=(256,256,3),activation='relu'))

cnn.add(MaxPool2D(pool_size=(2,2)))

cnn.add(BatchNormalization(axis = 1))

cnn.add(Dropout(0.2))

#second layer:
cnn.add(Conv2D(32,kernel_size=(4,4),activation = 'relu',input_shape=(256,256,3)))

cnn.add(MaxPool2D(pool_size=(2,2)))

cnn.add(BatchNormalization(axis = 1))

cnn.add(Dropout(0.2))

#third layer:
cnn.add(Conv2D(64,kernel_size=(3,3),activation='relu',input_shape=(256,256,3)))

cnn.add(MaxPool2D(pool_size=(2,2)))

cnn.add(BatchNormalization(axis = 1))

cnn.add(Dropout(0.2))

#fourth layer:
cnn.add(Conv2D(96,kernel_size=(3,3),activation='relu',input_shape=(256,256,3)))

cnn.add(MaxPool2D(pool_size=(2,2)))

cnn.add(BatchNormalization(axis = 1))



#flatten
cnn.add(Flatten())
cnn.add(Dropout(0.2))

cnn.add(Dense(units=512,activation='relu'))
cnn.add(Dense(units=512,activation='relu'))

cnn.add(Dense(units=196,activation='sigmoid'))

cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#end cnn model 

cnn.summary()

                                             
                                   
#Fit
his = cnn.fit_generator(train_data,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data = test_data,
                        validation_steps=nb_validation_samples//batch_size)
#his.history return {}
#end fit
results = pd.DataFrame.from_dict(his.history)
epoch_seires = pd.Series(range(0,30),name='epochs')
results = pd.concat([epoch_seires,results],axis=1)

#predict the image


pic = Image.open(pwd+"/car_data/train/Audi S4 Sedan 2007/00159.jpg")
display(pic)
def find_brand(dir=pwd+"/car_data/test"):
    classes = os.listdir(dir)
    classes.sort()
    classes_index = {classes[i]:i for i in range(len(classes))}
    return classes,classes_index
#test

find_brand()
#test
find_brand(pwd+"/car_data/test")




































