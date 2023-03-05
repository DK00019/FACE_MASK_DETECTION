#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt


# In[13]:


path='E:\py\dataanalytics\lord.jpg'
img = cv2.imread(path)


# In[14]:


img.shape #this gives the dimensions here height is 720,width is 1280 and color r,b,g so 3


# In[15]:


#here the color values that is rbg values are stored in arrays
img[0] #this gives an array of rbg values


# In[16]:


plt.imshow(img) #this is to pictorize our image but using matplotlib visualization the image will not have rbg colors


# In[19]:


#cap = cv2.VideoCapture(0)
 
#while(True):
#    ret, frame = cap.read()
 #   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
  #  cv2.imshow('frame',gray)
   # if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

#cap.release()
#cv2.destroyAllWindows()


# In[20]:


while True:
    cv2.imshow('result', img)
    if cv2.waitKey(2) == 27:
        break
cv2.destroyAllWindows()
#waitkey func waits for a key event infinitely , 2ms is delay because the os has a minimum time between switching threads as it performs multiple tasks
#27 is the ascii key value of escape key when ever we press esc key it will wait for 2ms and switch to another window


# #now lets look into how to carry on face detection
# #there is a popular algorithm known as viola jonnes algorithm for face detection
# viola jones algorthim:
# it is a object detection framework
# they divided this algorithm into 4 stages:
# 1)hara feature selection : this is nothing but there are some common features between the faces of all human being
# acc to it they gave some basic features like summation of pixels in black area - summation of pixels in white area
# if this is nearer to 1 there is a possibility of haar feature if it is close to zero then there is not a possibility of face availability
# so it makes a s;iding window from left to right throught out the image and calculate the haar feature if it is present then infers there is a face else there is no such face part this is how they are frame work works
# if the face is availabel they will return an array with x,y,height ,width of the image
# 2)creating an integral image
# 3)adaboost training
# 4)cascading classification

# they have already build a file known as haar cascade data:
# so if we search on google we will have a lot of xml files we can use this according to our requirement like to perform face detection eyes detection,license plate etc ...
# for our model we have to download haarcascade frontalface default.xml

# In[9]:


#now we have to load the classifier file 
haar_data = cv2.CascadeClassifier('E:\py\projects ML\mask detection\haarcascade_frontalface_default.xml') #this code helps to load the file and we have to pass the file name as arg


# In[10]:


#with this loaded features we could now be able to detect the faces
#now we use the inbuilt multiscale method to detect haar feature in our image
haar_data.detectMultiScale(img)
#the output is an array which consists of x,y,width and height of the face detected


# In[ ]:


# note : enter shift+tab to know about the arg of the func
#we will draw a rectangle over the face
#cv2.rectangle(img,(x,y),(w,h),(b,g,r),border_thickness) general code
#to do so we have copied the above code and pasted here
while True:
    faces = haar_data.detectMultiScale(img) #inorder to draw rect we have to detect the face first
    for x,y,w,h in faces:
        cv2.rectangle(img, (x,y), (x+w ,y+h), (255,0,255), 4)
        cv2.imshow('result', img)
        if cv2.waitKey(2) == 27:
            break
cv2.destroyAllWindows()


# In[ ]:


#now we know to detect the faces next we have to know whether they are wearing mask or not
#so we have to collect data now we have to start the camera
capture = cv2.VideoCapture(0) #here 0 is the default value for our camera #1 for external camera if we have installed and we put this data in a var
while True :
    flag, img = capture.read() #read will return two var one is flag and another one is img #this flag var will be either true or false it will give true if ur camera is working false if camera id not working
    if flag:
        faces = haar_data.detectMultiScale(img) #this is if camera is working we are gng to detect the face of the img
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w ,y+h), (255,0,255), 4)
        cv2.imshow('result', img)
        if cv2.waitKey(2) == 27:
            break #this is copied from above code that is if there is face it will draw a rectangle
capture.release() #this is used to release the camera that we are holiding
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




