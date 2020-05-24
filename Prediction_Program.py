#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Convolution2D,  MaxPooling2D, Flatten, Dense


# In[2]:


from keras.models import Sequential


# In[3]:


model=Sequential()


# In[4]:


model.add(Convolution2D(filters=32,
                        kernel_size=(3,3),activation='relu',
                        input_shape=(224,224,3)))


# In[5]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[6]:


model.add(Flatten())


# In[7]:


model.add(Dense(units=128,activation='relu'))


# In[8]:


model.add(Dense(units=3,activation='softmax'))


# In[9]:


model.summary()


# In[10]:


model.compile(optimizer='adam',loss='categorical_crossentropy',
             metrics=['accuracy'])


# In[11]:


from keras.preprocessing.image import ImageDataGenerator

train_data_dir = '/mnt/face_data/train/'
validation_data_dir = '/mnt/face_data/test/'

# Let's use some data augmentaiton 
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=45,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
# set our batch size (typically on most mid tier systems we'll use 16-32)
batch_size = 16
 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(224,224),
        batch_size=batch_size,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(224,224),
        batch_size=batch_size,
        class_mode='categorical')


history = model.fit_generator(
    train_generator,
    steps_per_epoch = 30, 
    epochs = 1,
    #callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = 30)


# In[12]:


model.save('Predict_me.h5')


# In[13]:


score=model.evaluate(validation_generator, verbose=0)
accuracy=float(score[1]*100)
#print("Accuracy: %.2f%%" %float(score[1]*100))
print("Accuracy:", accuracy)


# In[14]:


from keras.models import load_model

classifier = load_model('Guess_who_am_I.h5')


# In[15]:


import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

Face_detection_dict={ "[0]":"Mumma",
                        "[1]":"Palak",
                         "[2]":"Pranav" }

Face_detection_dict_n={ "n0":"Mumma",
                        "n1":"Palak",
                         "n2":"Pranav" }


def draw_test(name, pred, im):
    human = Face_detection_dict[str(pred)]
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100 ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, human, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    cv2.imshow(name, expanded_image)

def getRandomImage(path):
    """function loads a random images from a random folder in our test path """
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0,len(folders))
    path_class = folders[random_directory]
    #print("Class - " + Face_detection_dict_n[str(path_class)])
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    return cv2.imread(file_path+"/"+image_name)    

for i in range(0,3):
    input_im = getRandomImage("face_data/test/")
    input_original = input_im.copy()
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,224,224,3) 
    
    # Get Prediction
    res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)

    # Show image with predicted class
    draw_test("Guess who am I?", res, input_original) 
    cv2.waitKey(0)

cv2.destroyAllWindows()
