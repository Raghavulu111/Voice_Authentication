# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 17:17:49 2023

@author: Mahaboob Basha
"""

from scipy.io import wavfile
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt 
import numpy as np
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout
import streamlit as st
from scipy.signal import lfilter
from keras.layers import Conv1D, MaxPool1D, Flatten, Input
from keras.models import Model
from keras.layers import Dense
import pickle
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from cryptography.fernet import Fernet
from tkinter.filedialog import askopenfilename

# ===
# GETTING INPUT


file_up2 = askopenfilename()

print("Output of Read function is ")
tx1 = open("myfile.txt","r+") 

message = tx1.read()
    
key = Fernet.generate_key()
 
# Instance the Fernet class with the key
 
fernet = Fernet(key)
 
# then use the Fernet class instance
# to encrypt the string string must
# be encoded to byte string before encryption
encMessage = fernet.encrypt(message.encode())


file_up = askopenfilename()


samplerate, data = wavfile.read(file_up)
# data = data[0:1000]
data_2 = data.astype('int')
#    st.line_chart(data_2)

#import vlc
#p = vlc.MediaPlayer("sample.mp3")


#import playsound

#import pygame
#pygame.mixer.init()
#pygame.mixer.music.load(filename)

#playsound.playsound(filename)


plt.plot(data)
plt.title('ORIGINAL Data')
plt.show()

# PRE-PROCESSING

mu, sigma = 0, 500

plt.plot(data, linewidth=0.4, linestyle="-", c="b")  # it include some noise
plt.title('PRE-PROCESSED Data')
plt.show()



n = 15  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1
yy = lfilter(b,a,data)

#    st.line_chart(yy)

plt.plot(yy, linewidth=0.4, linestyle="-", c="b")  # smooth by filter
plt.title('FILTERED Data')
plt.show()

MN_val = np.mean(yy)

ST_val = np.std((yy))

VR_val = np.var((yy))

Min_val = np.min(yy)

Max_val = np.max(yy)


MN_val1 = np.mean(data)

ST_val1 = np.std((data))

VR_val1 = np.var((data))

Min_val1 = np.min(data)

Max_val1 = np.max(data)


Features = [MN_val,ST_val,VR_val,Min_val,Max_val,MN_val1,ST_val1,VR_val1,Min_val1,Max_val1]

#    st.text(Features)

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt


test_data1 = os.listdir('Dataset2/yes/')
test_data2 = os.listdir('Dataset2/no/')

dot= []
labels_target = []
grayscale_img1 = []
Features1 = []
for img in test_data1:
    
    try:
        samplerate, data = wavfile.read('Dataset2/yes/'+img)
        n = 15  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        yy = lfilter(b,a,data)
#            st.line_chart(yy)
        MN_val = np.mean(yy)
        ST_val = np.std((yy))
        VR_val = np.var((yy))
        Min_val = np.min(yy)
        Max_val = np.max(yy)
         
        MN_val1 = np.mean(data)
        ST_val1 = np.std((data))
        VR_val1 = np.var((data))
        Min_val1 = np.min(data)
        Max_val1 = np.max(data)
        Features1 = [MN_val,ST_val,VR_val,Min_val,Max_val,MN_val1,ST_val1,VR_val1,Min_val1,Max_val1]

        dot.append(np.array(Features1))
        labels_target.append(1)
        
    except:
        None
        
for img in test_data2:
    
    try:
        samplerate, data = wavfile.read('Dataset2/no/'+img)
        n = 15  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        yy = lfilter(b,a,data)
#            st.line_chart(yy)
        MN_val = np.mean(yy)
        ST_val = np.std((yy))
        VR_val = np.var((yy))
        Min_val = np.min(yy)
        Max_val = np.max(yy)
         
        MN_val1 = np.mean(data)
        ST_val1 = np.std((data))
        VR_val1 = np.var((data))
        Min_val1 = np.min(data)
        Max_val1 = np.max(data)
        Features1 = [MN_val,ST_val,VR_val,Min_val,Max_val,MN_val1,ST_val1,VR_val1,Min_val1,Max_val1]
        
        dot.append(np.array(Features1))
        labels_target.append(2)
        
    except:
        None
        
    
# with open('Trainfea1.pickle', 'rb') as f:
#     Train_features = pickle.load(f)

Train_features = dot
#    st.text(Train_features)
   
# Labels = np.arange(0,100)

# Labels[0:25] = 1
# Labels[25:50] = 2
# Labels[50:75] = 3
# Labels[75:100] = 4

#Label[101:150] = 3
#Label[151:200] = 4
#
#Label[201:250] = 5
#Label[251:300] = 6
#
#Label[301:350] = 7
#Label[351:400] = 8
#
#Label[401:450] = 9
#Label[451:500] = 10


clf = svm.SVC()
clf.fit(Train_features, labels_target)    

Class = clf.predict([Features])



neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(Train_features, labels_target)

Class_knn = neigh.predict([Features])

    
#os.system("program_name") 
# To open any program by their name recognized by windows

# OR

# Open any program, text or office document


if int(Class) == 1:
    print('=================')
    print('Recognized as - Yes')
    print('=================')
    

    
    # os.startfile("https://www.google.com/") 
    
    # os.startfile("Sample Folder/") 
    message1 = encMessage
    decMessage = fernet.decrypt(message1).decode()
    print(decMessage)
    
elif int(Class) == 2:
    print('=================')
    
    print('Recognized as - "No"')
    print('=================')
    


 
#    
from sklearn.metrics import accuracy_score

Predicted = clf.predict(Train_features)
Acc = accuracy_score(labels_target, Predicted)

print('----- Classification Accuracy ------')

print('====================================')

print(' Accuracy = ',Acc*100,' %')
print('====================================')


    