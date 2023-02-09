import cv2
import os
from PIL import Image
import tensorflow as tf
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout,Flatten, Dense
from keras.utils import to_categorical
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split


#To have a list of all the differents images that are not tumor
image_directory = 'Good-dataset/'
no_tumor_images = os.listdir(image_directory+'Healthy/')
yes_tumor_images = os.listdir(image_directory+'Brain Tumor/')

dataset=[]
label = []
INPUT_SIZE = 64
#To read the images that are jpeg in our no folder
for i, image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'): #to read only the jpg images
        image=cv2.imread(image_directory+'Healthy/'+image_name)
        image = Image.fromarray(image,'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))   #Add the images into our dataset
        label.append(0) #Meaning that for our prediction if it's a zero then it's not a brain tumor
        
        
        

#To read the images that are jpeg in our no folder
for i, image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'): #to read only the jpg images
        image=cv2.imread(image_directory+'Brain Tumor/'+image_name)
        image = Image.fromarray(image,'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))   #Add the images into our dataset
        label.append(1) #Meaning that for our prediction if it's a zero then it's not a brain tumor        
#print(len(label)) To see how many images is in our dataset

#Convert dataset into numpy array
dataset=np.array(dataset)
label=np.array(label)


#Do the split between train and test set
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2,random_state=0)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)


# Do not need this if your model is a dense=1 and sigmoid model
y_train = to_categorical(y_train, num_classes=2) # We do have only two classes
y_test = to_categorical(y_test,num_classes=2) #same as above

# BUILD OUR MODEL


model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


#Need now to make all the images in one factor
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2)) #It's because it is a binary classification but you could make dense=2 if it was a categorical classification
model.add(Activation('softmax')) # Again, if you use categorical entropy then put dense=2 and activation=softmax vs sigmoid for the dense=1


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train, batch_size=16, verbose=1, epochs=10, validation_data= (x_test,y_test),shuffle=False)

model.save('BrainTumor10EpochsCategorical.h5')  # TODO LOOK WHAT IS EPOCHS ( we used CNN for our model)




