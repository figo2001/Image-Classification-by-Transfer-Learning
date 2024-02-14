#!/usr/bin/env python
# coding: utf-8

# ## 1) Transfer Learning with MobileNet V2
# 
# #### Introduction
# In this notebook, I demonstrate the application of transfer learning using MobileNet V2, a lightweight convolutional neural network architecture, for a specific task. Transfer learning with MobileNet V2 is particularly useful when working with limited computational resources or deploying models on mobile or edge devices.
# 
# #### MobileNet V2 Overview
# MobileNet V2 is an efficient convolutional neural network architecture designed for mobile and embedded vision applications. It is an extension of the original MobileNet architecture, featuring improved performance and reduced computational complexity. MobileNet V2 achieves a good balance between model size, accuracy, and speed, making it well-suited for various computer vision tasks.
# 
# #### Transfer Learning Approach
# In this notebook, I adopt a fine-tuning approach for transfer learning with MobileNet V2:
# 1. **Loading Pre-Trained Model**: I utilize the pre-trained MobileNet V2 model, which has been pre-trained on the ImageNet dataset. The pre-trained weights capture generic image features that can be useful for a wide range of tasks.
# 2. **Model Adaptation**: I modify the MobileNet V2 architecture by replacing the fully connected classification layers with new layers tailored to the specific task at hand. These new layers are initialized randomly and will be trained on the target dataset.
# 3. **Fine-Tuning**: I fine-tune the adapted MobileNet V2 model on the target dataset. During fine-tuning, the weights of the entire network are updated using gradient descent to minimize the loss function on the target dataset.
# 4. **Evaluation**: I evaluate the performance of the fine-tuned MobileNet V2 model on a separate validation dataset. Metrics such as accuracy, precision, recall, and F1 score may be computed to assess the model's performance.
# 
# #### Implementation Details
# - **Framework**: I use TensorFlow/Keras for implementing the transfer learning pipeline with MobileNet V2.
# - **Dataset**: The dataset used for fine-tuning MobileNet V2 may vary depending on the specific task. It could be a custom dataset or a publicly available dataset relevant to the task.
# - **Hyperparameters**: Hyperparameters such as learning rate, batch size, and number of epochs are crucial for fine-tuning MobileNet V2 effectively. These hyperparameters may be tuned through experimentation to achieve optimal performance.
# 
# #### Conclusion
# Transfer learning with MobileNet V2 offers a practical and efficient solution for various computer vision tasks, especially in scenarios with limited computational resources or deployment constraints. By leveraging the pre-trained MobileNet V2 model and fine-tuning it on a target dataset, we can achieve good performance with reduced training time and computational cost.
# 
# #### References
# - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
# - [TensorFlow Hub: MobileNet V2](https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4)
# - [Transfer Learning with TensorFlow](https://www.tensorflow.org/tutorials/images/transfer_learning)

# ----------------------------------------------------------------------------------------------------------------------------------------

# ## 2) Extracting Dataset using Kaggle API

# **installing the kaggle library**

# In[ ]:


get_ipython().system('pip install kaggle')


# ## 3) Importing the Dog vs Cat Dataset from Kaggle

# #### by using kaggle API

# - Create Directory, If the .kaggle directory doesn't exist, you need to create it before copying the file
# - Ensure that you're using the correct file path when copying 'kaggle.json'.
# - After copying the file, ensure that you set the correct permissions

# In[ ]:


get_ipython().system('kaggle competitions download -c dogs-vs-cats')


# In[ ]:


get_ipython().system('ls')


# #### Extracting the compressed dataset

# In[ ]:


from zipfile import ZipFile

dataset='/kaggle/working/dogs-vs-cats.zip'

with ZipFile(dataset, 'r') as zip:
    zip.extractall()
    print('The dataset is extracted')


# **Extracting the compressed dataset**

# In[ ]:


from zipfile import ZipFile

dataset='/kaggle/working/train.zip'

with ZipFile(dataset, 'r') as zip:
    zip.extractall()
    print('The dataset is extracted')


# **Counting the number of files in train folder**

# In[ ]:


import os

path, dirs, files = next(os.walk('/kaggle/working/train'))
file_count = len(files)
print('Number of images: ', file_count)


# ### Printing the name of images

# In[ ]:


file_name=os.listdir('/kaggle/working/train')
print(file_name)


# ## 4) Importing the dependencies

# In[ ]:


import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
import cv2


# #### a) Showing the dog images

# In[ ]:


img=mpimg.imread('/kaggle/working/train/dog.9850.jpg')
imgplt=plt.imshow(img)

plt.show()


# #### b) Showing Cat images

# In[ ]:


img=mpimg.imread('/kaggle/working/train/cat.8771.jpg')
imgplt=plt.imshow(img)

plt.show()


# - #### Now we have to resize all of the cat and dog images into the same resolution

# ### In this scenario, only using first three letters of the cat and dog images, it's a huge dataset -- So it's a good practice.

# In[ ]:


file_name=os.listdir('/kaggle/working/train')

for i in range (5):
    name=file_name[i]
    print(name[0:3])


# ## 5) Counting how many Dog and Cat images are in present here.

# In[ ]:


file_name=os.listdir('/kaggle/working/train')

dog_count=0
cat_count=0

for img_file in file_name:

    name=img_file[0:3]

    if name=='dog':
        dog_count +=1
    else:
        cat_count +=1

print('Number of Dog images: ',dog_count)
print('Number of Cat images: ',cat_count)


# ### Resizing the images

# **Creating the directory for resized images**

# In[ ]:


os.mkdir('/kaggle/working//image_resized')


# In[ ]:


original_folder='/kaggle/working/train/'
resized_folder='/kaggle/working//image_resized/'

for i in range(2000):

    filename=os.listdir(original_folder)[i]
    img_path=original_folder+filename


    # Resizing the images

    img=Image.open(img_path)
    img=img.resize((224,224))
    img=img.convert('RGB')


    newImgPath=resized_folder+filename
    img.save(newImgPath)


# #### Viewing the resized dog image

# In[ ]:


img=mpimg.imread('/kaggle/working/image_resized/dog.8766.jpg')
imgplt=plt.imshow(img)

plt.show()


# #### Viewing the resized Cat image

# In[ ]:


img=mpimg.imread('/kaggle/working/image_resized/cat.4892.jpg')
imgplt=plt.imshow(img)

plt.show()


# ## 6) Creating Labels for resized images of dogs and cats
# 
# - **cat --> 0**
# - **dog --> 1**

# ### Creating a for loop for assign labels

# In[ ]:


filename=os.listdir('/kaggle/working/image_resized/')

labels=[]

for i in range(2000):

    file_name=filename[i]
    label=file_name[0:3]


    if label=='dog':
        labels.append(1)

    else:
        labels.append(0)


# - **Showing the actual files.**

# In[ ]:


print(filename[0:5])

print(len(filename))


# - **Total the number of labels**

# In[ ]:


print(labels[0:5])

print(len(labels))


# - **Counting the images of Dog and Cats out of 2000 images**

# In[ ]:


values, counts=np.unique(labels, return_counts=True)

print(values)
print(counts)


# ## 7) Converting all the resized images to numpy arrays

# In[ ]:


import cv2
import glob


# In[ ]:


image_directory='/kaggle/working/image_resized/'
image_extension=['png','jpg']

files=[]

# checking the files if it is jpg or png then convert into numpy array
[files.extend(glob.glob(image_directory + "*." + e)) for e in image_extension]

# convert into numpy array and convert into a single numpy array
dog_cat_images=np.asarray([cv2.imread(file) for file in files ])


# In[ ]:


print(dog_cat_images)


# - **showing the type**

# In[ ]:


type(dog_cat_images)


# - **showing the shape**

# In[ ]:


print(dog_cat_images.shape)


# ## 8) Train Test Split

# In[ ]:


X = dog_cat_images
y = np.asarray(labels)


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=2)


# In[ ]:


print(X.shape,X_train.shape,X_test.shape)


# - **1600 --> Training images**
# - **400 --> Test images**

# ## 9) Scalling the data

# In[ ]:


X_train_scaled=X_train/255

X_test_scaled=X_test/255


# In[ ]:


print(X_train_scaled)


# # 10) Building the Neural Network - MobileNet V2

# In[ ]:


import tensorflow as tf
import tensorflow_hub as hub


# In[ ]:


mobilenet_model = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'

pretrained_model = hub.KerasLayer(mobilenet_model, input_shape=(224,224,3), trainable=False)


# - #### i) Building the model

# In[ ]:


num_of_classes=2

model=tf.keras.Sequential([

    pretrained_model,
    tf.keras.layers.Dense(num_of_classes)
])

# Showing the model summary
model.summary()


# - #### ii) Compiling the model

# In[ ]:


model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ['acc']
)


# - #### iii) Training the model

# In[ ]:


history=model.fit(X_train_scaled, y_train,epochs=5)


# #### iv) Model Evaluation

# In[ ]:


score, acc = model.evaluate(X_test_scaled, y_test)
print('Test Loss =', score)
print('Test Accuracy =', acc)


# ## 11) Predictive System

# - ### For Dog 🐶

# In[ ]:


# Take user input as image
input_img_path=input('Path of the image to be predicted')

# read this image and load into numpy array
input_image=cv2.imread(input_img_path)

# display the image
plt.imshow(input_image)

# later that resize the image
input_image_resize=cv2.resize(input_image,(224,224))

# scale the image
input_image_scaled=input_image_resize/255

# now reshaping the numpy array --> to tell the model that I am making predictions for only 1 image, that's why I am using 1 here.
image_reshaped=np.reshape(input_image_scaled, [1,224,224,3])

# pass this image_reshaped into input_prediction
input_prediction=model.predict(image_reshaped)

print(input_prediction)

# now pass into np.argmax and np.argmax is nothing but the prediction.
input_pred_label=np.argmax(input_prediction)

print(input_pred_label)

if input_pred_label==0:
    print('The image is cat ')

else:
    print('The image is dog')


# - ### For Cat🐈

# In[ ]:


# Take user input as image
input_img_path=input('Path of the image to be predicted')

# read this image and load into numpy array
input_image=cv2.imread(input_img_path)

# display the image
plt.imshow(input_image)

# later that resize the image
input_image_resize=cv2.resize(input_image,(224,224))

# scale the image
input_image_scaled=input_image_resize/255

# now reshaping the numpy array --> to tell the model that I am making predictions for only 1 image, that's why I am using 1 here.
image_reshaped=np.reshape(input_image_scaled, [1,224,224,3])

# pass this image_reshaped into input_prediction
input_prediction=model.predict(image_reshaped)

print(input_prediction)

# now pass into np.argmax and np.argmax is nothing but the prediction.
input_pred_label=np.argmax(input_prediction)

print(input_pred_label)

if input_pred_label==0:
    print('The image is cat ')

else:
    print('The image is dog')


# # Thank You!
# 
# Thank you for taking the time to explore my notebook! I hope you found it informative and insightful.

# ----------------------------------------------------------------------------------------------------------------------------------------

# In[ ]:




