#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.layers import Convolution2D,  MaxPooling2D, Flatten, Dense


# In[3]:


from keras.models import Sequential


# In[4]:


model=Sequential()


# In[5]:


import numpy as np

import hyper_parameters as hp
input_shape=(224,224,3)

def addCRPs(i):
    if i==1:  # 1 Conv and 1 Pool
        model.add(Convolution2D(filters=hp.no_of_filters,
        kernel_size=(hp.kernel_size,hp.kernel_size),activation='relu',
        input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(hp.pool_size,hp.pool_size)))
    if i ==2:  #2 Conv and 1 Pool
        model.add(Convolution2D(filters=hp.no_of_filters*2,
        kernel_size=(hp.kernel_size+2,hp.kernel_size+2),activation='relu',
        input_shape=input_shape))
        model.add(Convolution2D(filters=hp.no_of_filters,
        kernel_size=(hp.kernel_size,hp.kernel_size),activation='relu',
        input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(hp.pool_size,hp.pool_size)))
    elif i ==3:  # 2Conv and 2Pool
        model.add(Convolution2D(filters=hp.no_of_filters*2,
        kernel_size=(hp.kernel_size+2,hp.kernel_size+2),activation='relu',
        input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(hp.pool_size+2,hp.pool_size+2)))
        
        model.add(Convolution2D(filters=hp.no_of_filters,
        kernel_size=(hp.kernel_size,hp.kernel_size),activation='relu',
        input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(hp.pool_size,hp.pool_size)))
    elif i ==4:  #3 Conv and 1 Pool
        model.add(Convolution2D(filters=hp.no_of_filters*4,
        kernel_size=(hp.kernel_size+4,hp.kernel_size+4),activation='relu',
        input_shape=input_shape))
        
        model.add(Convolution2D(filters=hp.no_of_filters*2,
        kernel_size=(hp.kernel_size+2,hp.kernel_size+2),activation='relu',
        input_shape=input_shape))
        
        model.add(Convolution2D(filters=hp.no_of_filters,
        kernel_size=(hp.kernel_size,hp.kernel_size),activation='relu',
        input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(hp.pool_size,hp.pool_size)))

        
addCRPs(hp.i)      


# In[6]:


#np.expand_dims()
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=3,activation='softmax'))

model.summary()


# In[7]:


model.compile(optimizer='adam',loss='categorical_crossentropy',
             metrics=['accuracy'])


# In[8]:


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
    epochs = 2,
    #callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = 30)


# In[9]:


model.save('Predict_me.h5')


# In[10]:


print(model.history.history)


# In[11]:


score=model.evaluate(validation_generator, verbose=0)
accuracy=float(score[1]*100)
print(f"accuracy {accuracy}%")


# In[14]:


import os
os.system("cp hyper_parameters.py old_hyper_parameters.py")


if accuracy > 90:
    #system("echo True accuracy={}% > Accuracy.txt".format(accuracy))
    system("echo 'True' accuracy={}% > /Accuracy.txt".format(accuracy))
else:
    #system("echo False accuracy={}% > Accuracy.txt".format(accuracy))
    system("echo 'False' accuracy={}% > /Accuracy.txt".format(accuracy))


# In[15]:


import os

#print(os.path "Accuracy.txt")
print (os.path.abspath("Accuracy.txt"))


# In[16]:


from keras.models import load_model

classifier = load_model('Predict_me.h5')


# In[17]:


# Code to send email

import smtplib, ssl

port = 465 #For SSL
smtp_server ="smtp.gmail.com"
sender_email="iampalakjain01@gmail.com"    #Sender's Mail Address
receiver_email="itspalak19@gmail.com"      #Receiver's Mail Address
password="xecbeupbulzfwpos"
if accuracy > 90:
    message="""    Subject: Report | Prediction Program
    
    CONGRATULATIONS! 
    Your code achieved{}% accuracy.""".format(accuracy)
else:
    message="""    Subject: Report | Prediction Program
    
    Train Again!
    Your code got {}% accuracy.""".format(accuracy)
    
context=ssl.create_default_context()
with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
    server.login(sender_email,password)
    server.sendmail(sender_email, receiver_email, message)
    
    
    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




