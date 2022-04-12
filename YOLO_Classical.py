# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:38:50 2022

@author: PRATHAMESH
"""
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.datasets import mnist
import numpy as np
import cv2
import matplotlib.pyplot as plt
#%matplotlib inline
import math
import pandas as pd
import os
from os import listdir
#YOLO

#RBM
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

import numpy as np
import os
from skimage.io import imread_collection
from skimage.io import concatenate_images


a=np.matrix([])


help(np.matrix)
v0=np.zeros((1,8))
h0=np.zeros((1,8))
v1=np.zeros((1,8))
h1=np.zeros((1,8))
print(a)

a[0][3]=90

# object_exits
# x
# y
# height
# width
# class

#Weak classifier:
#Input to the RBM is 64 * 64 image
#Output is 8 *1 array
#Qboost


# folder_dir = "D:\Abhiyantrik Prathamesh\PTM MIM 1922\SEM SAHAVA\Research Project\MoCA\JPEGImages"
# for image_sub_dir in os.listdir(folder_dir):
#  for images   in  os.listdir(folder_dir+"/"+image_sub_dir):
#     # check if the image ends with png
#     if (images.endswith("00095.jpg")):
#         print(images)
        

 # image=cv2.imread(folder_dir+"/"+image_sub_dir+"/"+images)
 # cv2.imshow("w", image)
 # cv2.waitKey(0)       
 

#  df = pd.DataFrame(grayscale)
#  import cv2
#  df.to_csv(r'D:\Abhiyantrik Prathamesh\PTM MIM 1922\SEM SAHAVA\Research Project\Exported_Image.csv')
# cv2.imshow('Original',image)
# grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Grayscale', grayscale)
# cv2.waitKey(0)  

grayscale.shape



#X_train.head()
X_train = (X_train - np.min(X_train, 0)) / (np.max(X_train, 0) + 0.0001)
 # rescale to (0, 1)
 
from sklearn.neural_network import BernoulliRBM
rbm = BernoulliRBM(n_components=100, learning_rate=0.01, random_state=42, verbose=True)
rbm.fit(X_train)


def gen_mnist_image(X):
    return np.rollaxis(np.rollaxis(X[0:200].reshape(20, -1, 28, 28), 0, 2), 1, 3).reshape(-1, 20 * 28)

xx = X_train[:40].copy()
for _ in range(1000):
    for n in range(40):
        xx[n] = rbm.gibbs(xx[n])
        
import matplotlib.pyplot as plt
plt.figure(figsize=(512,512))
plt.imshow(gen_mnist_image(xx), cmap='gray')
plt.show()


plt.figure(figsize=(20, 20))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((10, 128)), cmap=plt.cm.RdBu,
               interpolation='nearest', vmin=-2.5, vmax=2.5)
    plt.axis('off')






#load images in collection
folder_dir = "D:\Abhiyantrik Prathamesh\PTM MIM 1922\SEM SAHAVA\Research Project\MoCA\JPEGImages"

image_sub_dir="wolf"

imgs=imread_collection("D:\Abhiyantrik Prathamesh\PTM MIM 1922\SEM SAHAVA\Research Project\MoCA\JPEGImages\arabian_horn_viper\\"+"/*.jpg")
len(imgs)
for image_sub_dir in os.listdir(folder_dir):
 imgs = imread_collection(folder_dir+"/"+image_sub_dir+'/*.jpg')   
 for n in range(0,(int) (len(imgs)/5)+1):       
      cv2.imwrite(f'D:\Abhiyantrik Prathamesh\PTM MIM 1922\SEM SAHAVA\Research Project\MoCA\ALLTogetherImages\{image_sub_dir}_image_{n}.jpg',imgs[n])
   
 for images   in  os.listdir(folder_dir+"/"+image_sub_dir):
    # check if the image ends with png
     if (images.endswith("0.jpg") or images.endswith("5.jpg")):
        
        imgs = imread_collection(folder_dir+"/"+image_sub_dir+'/*.jpg')
   
folder_dir = "D:\Abhiyantrik Prathamesh\PTM MIM 1922\SEM SAHAVA\Research Project\MoCA"    
image_sub_dir='ALLTogetherImages'


    
imgs = imread_collection(folder_dir+"/"+image_sub_dir+'/*.jpg')
    

#Convert Image to gray scale and resize to 128 * 128
imgs = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in imgs]    
imgs_new = [x for x in imgs] 
from skimage.transform import resize
imgs = [resize(x,(128,128), mode='constant', anti_aliasing=False) for x in imgs]

ic=concatenate_images(imgs)

print(len(imgs))


print(ic)
print(imgs)
ic.shape
        


grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
flatImgs = [x.flatten() for x in imgs]
flatImgsnew = [x.flatten() for x in imgs]

print(imgs)


from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

rbm = BernoulliRBM(learning_rate=.01, n_iter=20, n_components=150)
print(rbm)
logistic = LogisticRegression(solver='lbfgs', 
                 max_iter=10000,C=6000, multi_class='multinomial')
linear =LinearRegression()

#Combine the two into a Pipeline
rbm_features_classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])



#Create a target variable: an ID between 1 and 15 for each of the 15 subjects
#Y = [[id for pose in range(1,12)] for id in range(1,6)]
Y=anno
#Flatten the 'list of lists' into a 2D list
#Y = [num for sublist in Y for num in sublist]

print(Y)
anno=pd.read_csv("Annotations_edited2.csv")
anno.info()

anno3=anno[['spatial_coordinates','file_list','Start_Point','End_point','Length','Breadth']]


anno4=anno3['spatial_coordinates'].str.split(',')

anno4.head()


anno5=

anno4[0][0]
anno4[0][1]
anno4[0][2]
anno4[0][4]

anno3[]=anno4[:1]
anno3[]=anno4[:2]
anno3[]=anno4[:3]
anno3[].head()

anno3.head(1)
anno3.info()





for 




anno3=anno[anno.spatial_coordinates,anno.file_list]

print(anno2[1])
anno5.info()
print(anno3.shape)
 
anno5.head()

  

for i in ['1','2','3','4','6','7','8','9']:
 anno3=anno3.drop(anno3[anno3['file_list'].str.contains(i+'.jpg')].index)


anno=anno.tail(96)
Y=anno
print(Y.shape)
import pandas as pd
import numpy as np
X=flatImgs.copy()
X_train, X_test=train_test_split(X, test_size=0.3, random_state=0)

Y_train, Y_test = train_test_split(Y, test_size=0.3, random_state=0)
print(X_train)
 # exclude the target
 
rbm_features_classifier.fit(flatImgs, Y)
#rbm.fit(flatImgs, Y)

Y_pred = rbm_features_classifier.predict(X_test)

print(Y_pred)
#Y_pred=rbm.predict(flatImgs)
from sklearn import metrics
metrics.classification_report(Y, Y_pred)
print(Y_pred)
import matplotlib.pyplot as plt
from skimage.io import imshow

plt.figure(figsize=(15, 15))
for i, comp in enumerate(rbm.components_[:150]):
    plt.subplot(15, 10, i + 1)
    plt.imshow(comp.reshape((128, 128)), cmap=plt.cm.gray_r,
        interpolation='nearest')    
    plt.xticks(())
    plt.yticks(())
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
plt.show()



# Selected features for closer examination
toShow = [104,116,84]

plt.figure(figsize=(16, 10))
for i, comp in enumerate(toShow):
    plt.subplot(1,3,i+1)
    plt.imshow(rbm.components_[comp].reshape((128, 128)), cmap=plt.cm.gray_r,
        interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.show()

Y_pred[0][4]

#Show Rectangle
#[2,472.607,289.194,132.227,154.408]
  
# Start coordinate, here (5, 5)
# represents the top left corner of rectangle

#isrt_point = (262,314)
start_point = (472,289)  
# Ending coordinate, here (220, 220)
# represents the bottom right corner of rectangle
l=472+132
b=289+154
#nd_point = (l,b)
end_point =(l,b)
# Blue color in BGR
color = (0, 255, 0)



# '[2,262.314,244.411,231.956,189.146]']
  
# Line thickness of 2 px
thickness = 2
 

# Using cv2.rectangle() method
# Draw a rectangle with blue line borders of thickness of 2 px
img_rect=flatImgsnew[2]

#img_rect=cv2.cvtColor(img_rect,cv2.COLOR_GRAY2RGB) 
img_rect = cv2.rectangle(imgs_new[1], start_point, end_point, color, thickness)
plt.imshow(img_rect.reshape((1280, 720,3)))
cv2.imshow("Khi", img_rect)

cv2.waitKey(0)     
#plt.figure(figsize=(15, 15))
# Displaying the image 

plt.imshow(flatImgs[97].reshape((128, 128)) ,cmap='gray')
  
img_rect = cv2.rectangle(img_rect, start_point, end_point, color, thickness)      




alpha=1.0
w_posi_grad=tf.matmul(tf.transpose(v0),h0)
w_nega_grad=tf.matmul(tf.transpose(v1),h1)
CD=(w_posi_grad-w_nega_grad)/tf.    (tf.shape(v0)[0])
    
    