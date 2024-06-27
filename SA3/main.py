import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,BatchNormalization,Flatten,Input
from sklearn.model_selection import train_test_split
from cvzone.FaceDetectionModule import FaceDetector

path = "../Face-image-dataset"
images = []
ages = []
gender = []

detector = FaceDetector()

for img in os.listdir(path)[0:10]:
  try:
    if img!='.git':
      age = img.split("_")[0]
      genders = img.split("_")[1]
      img = cv2.imread(str(path)+"/"+str(img))
      img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      
      img, bbox = detector.findFaces(img, draw=False) 
      if bbox:
        X = bbox[0]['bbox'][0]
        Y = bbox[0]['bbox'][1]
        W = bbox[0]['bbox'][2]
        H = bbox[0]['bbox'][3]
        
        img = img[Y:Y+H, X:X+W]
        img = cv2.resize(img, (200, 200))
        
      images.append(img)
      ages.append(age)

  except Exception as e:
    print("Exception",e)

ages = np.array(ages,dtype=np.int64)
images = np.array(images)

training_images, testing_images, training_ages, testing_ages = train_test_split(images, ages)
print(":::::::::::::::::::::::::::::::::::::::::::::::::")
print("Number of training images = ", len(training_images))
print("Number of testing images = ", len(testing_images))
print("Number of training ages = ", len(training_ages))
print("Number of testing ages = ", len(testing_images))
print(":::::::::::::::::::::::::::::::::::::::::::::::::")

age_model = Sequential()

age_model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(200,200,3)))
age_model.add(MaxPool2D(pool_size=3, strides=2))

age_model.add(Conv2D(128, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))
              
age_model.add(Conv2D(256, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))

age_model.add(Conv2D(512, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))

age_model.add(Flatten())
age_model.add(Dropout(0.2))
age_model.add(Dense(512, activation='relu'))

age_model.add(Dense(1, activation='linear', name='age'))
              
age_model.compile(optimizer='adam', loss='mae')

print(age_model.summary()) 

history = age_model.fit(training_images, training_ages, validation_data=(testing_images, testing_ages), epochs=10)

age_model.save('age_model_50epochs.h5')

loss = history.history['loss']
epochs = range(1, len(loss) + 1)

# Create a val_loss variable and store history.history['val_loss'] in it
val_loss = history.history['val_loss']

plt.plot(epochs, loss, 'y', label='Training loss')

# Plot a line between epos and val_loss if red color and label it as Validation loss
plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



