# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 13:16:51 2019

@author: 14103
"""
#IMPORT LIBRARY
import numpy as np
import cv2

#INPUT IMAGE
img = cv2.imread('image/P5.png')
scale_percent = 50 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
cv2.imshow('a',resized)
#DILATE CONTOUR
kernel = np.ones((6,6), np.uint8)
img_dilation = cv2.dilate(resized, kernel, iterations=1)
#FIND EDGES IN IMAGE BY CANNY
edges = cv2.Canny(img_dilation,100,200)
im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.imshow('r', im2)
#CHOOSE FIRST CONTOUR
cnt = contours[0]
cv2.drawContours(resized, [cnt], 0, (0,255,0), 3)
X = cnt[:,0,0]
Y = cnt[:,0,1]
M = cv2.moments(cnt)
#FINDING CENTROID OF CONTOUR
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
#SYMBOL CONTOUR AND CENTROID
cv2.circle(resized, (cx, cy), 5, (255, 255, 255), -1)
cv2.putText(resized, "centroid", (cx - 25, cy - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#SHOW IMAGE
cv2.imshow('b',resized)
#CHANGE POLAR COORDINATE INTO CARTESIAN COORDINATE
x = X - np.mean(X)
y = Y - np.mean(Y)
r = np.sqrt(x**2+y**2)
t = -np.arctan2(y,x)

#USING MATPLOTLIB TO PLOT PICTURES
import matplotlib.pyplot as plt
from matplotlib import interactive
#PLOT RADIUS
plt.figure(1)
plt.plot(r)
interactive(True)
plt.show()


#PLOT DRAPE CONTOUR IN POLAR COORDINATE
plt.figure(2)
plt.plot(x,-y)
plt.axis('equal')
plt.show()

#PLOT DRAPE CURVE IN POLAR COORDINATE
plt.figure(3)
plt.polar(t, r, 'g.')
plt.show()

#LENGTH OF RADIUS
N = len(r)
index = np.linspace(0, N-1, num=N)

#APPLY SAVGOL FILTER TO REMOVE NOISE OF THE CURVE
from scipy.signal import savgol_filter
rhat = savgol_filter(r, 51, 3) # window size 51, polynomial order 4


#PLOT RADIUS REMOVED NOISE AND THE ORIGION RADIUS
plt.figure(4)
plt.plot(r)
plt.plot(rhat, color='red')
plt.show()

#FIND MINIMUM VALUE OF RHAT (REMOVED NOISE OF R)
for i in range(0, len(rhat)-1):
    if rhat[i] == np.min(rhat):
        global m
        m = i
#START RHAT AT THE MINIMUM OF RHAT       
rhat = np.roll(rhat,-m)
plt.figure(5)
plt.plot(rhat, color='red')
plt.show()


#FIND PEAKS
from scipy.signal import find_peaks
time_series = rhat
indices = find_peaks(time_series)[0]
plt.figure(6)
plt.plot(time_series)
plt.plot(indices, time_series[indices], "gx")
plt.show()
#FIND VALLEYS
time_series2 = rhat*(-1)
indices2 = find_peaks(time_series2)[0]
plt.figure(7)
plt.plot(time_series)
plt.plot(indices2, time_series[indices2], "rv")
plt.show()
#PLOT PEAKS AND VALLEYS
plt.figure(8)
plt.plot(time_series)
plt.plot(indices, time_series[indices], "gx")
plt.plot(indices2, time_series[indices2], "rv")
plt.show()
#DELETE THE PEAKS OF OUTLIER LOWER THAN MEAN OF TIME SERIES
print(indices)
index = []
print(len(indices)-1)
print(indices[len(indices)-1])
print(time_series[indices[len(indices)-1]])
print(np.mean(time_series))
for i in range(0,len(indices)):
    if time_series[indices[i]]<np.mean(time_series):
        index.append(i)
print('index peak',index)
print(indices)
for i in reversed(index):
    indices = np.delete(indices,i)
print(indices)

b = 0.48*(0.5*(np.max(time_series)+np.min(time_series))+np.max(time_series))

#DELETE THE VALLEYS OF OUTLIER HIGHER THAN HALF OF AMPLITUDE
index = []
print('b',b)
print('indices2', time_series[indices2])
for i in range(0,len(indices2)):
    if b<time_series[indices2[i]]:
        index.append(i)
print('index', index)
print(b)
print(time_series[indices2[index]])
print(indices2)
for i in reversed(index):
    indices2 = np.delete(indices2,i)

print(indices2)
print(np.mean(time_series[indices]))
#DELETE EXCESSES PEAKS
indice1 = []
if len(indices2)<len(indices):
    for j in range(0, len(indices2)):
        indice1.append(indices[j])
if len(indices2)<len(indices):
    sub = np.abs(indice1 - indices2)
    for i in range(0, len(sub)-1):
        if sub[i]<0.1*np.mean(sub):
            indices = np.delete(indices, i)
#PLOT PEAKS
plt.figure(9)
plt.plot(time_series)
plt.plot(indices, time_series[indices], "gx")
plt.plot(indices2, time_series[indices2], "rv")
plt.show()
print('1', indices)
print('peaks', len(indices), indices[0])
print('valleys', len(indices2), indices2[0])
#DELETE THE NOISE VALLEYS
if len(indices2)!=len(indices):
    if indices[0]>indices2[0]:
        if len(indices2)>len(indices):
            for i in range(0, len(indices)-1):
                if indices2[i]<indices[i]:
                    if indices2[i+1]<indices[i]:
                        if time_series[indices2[i]]<time_series[indices2[i+1]]:
                            indices2 = np.delete(indices2, i+1)
                        else:
                            indices2 = np.delete(indices2, i)
    
#DELETE THE NOISE PEAKS       
    elif indices[0]<indices2[0]:
        if len(indices2)<len(indices):
            i=0
            while(i<=len(indices2)-2):               
                if indices[i+1]<indices2[i]:
                    if time_series[indices[i]]<time_series[indices[i+1]]:
                        indices = np.delete(indices, i)
                    else:
                        indices = np.delete(indices, i+1)
                    i = i -1
                i = i+1
                

        
print('len(indices)', len(indices))
print('len(indices2)', len(indices2)) 
plt.figure(10)
plt.plot(time_series)
plt.plot(indices, time_series[indices], 'gx')
plt.plot(indices2, time_series[indices2], 'rv')
plt.show()
                     
#REMOVE NOISE OF VALLEYS
if len(indices) == len(indices2):
    if indices[0]<indices2[0]:
        for i in range(0, len(indices)-2):
            if indices[i]<indices2[i]:
                if indices2[i+1]<indices[i+1]:
                    if time_series[indices2[i]]<time_series[indices2[i+1]]:
                        indices2 = np.delete(indices2, i+1 )
                    else:
                        indices2 = np.delete(indices2, i)
           
if len(indices)==len(indices2):
    if indices[0]<indices2[0]:
        for i in range(0, len(indices2)-2):
            if indices[i+1]>indices2[i+1]:
                if time_series[indices2[i]]<time_series[indices2[i+1]]:
                    indices2 = np.delete(indices2, i+1)
                else:
                    indices2 = np.delete(indices2, i)
#PRINT INDEX OF PEAKS AND VALLEYS     
print('1' , indices) 
print('2', indices2)

#ADD THE FIRST VALLEY =  MINIMUM OF RHAT
a = 0    
if len(indices2)!=len(indices):
    if len(indices)>len(indices2):
        for indice in indices2:
            if np.min(time_series)==time_series[indice]:
                a=1
                
if a==0:
    indices2 = np.append(indices2, 0)
    indices2=np.sort(indices2)
                
  
else:
    print('False')  
    
plt.figure(11)
plt.plot(time_series)
plt.plot(indices, time_series[indices], 'gx')
plt.plot(indices2, time_series[indices2], 'rv')
plt.show()
    
#DELETE THE NOISE VALLEYS
if time_series[indices2[len(indices2)-1]]>np.mean(time_series[indices]):
        indices2 = np.delete(indices2, len(indices2)-1)
       
print(indices2)
print('peaks2', len(indices))
print('valley2', len(indices2))
x = indices[len(indices)-2]
y = indices2[len(indices2)-1]              
if x>y:
    if time_series[indices[len(indices)-1]]>time_series[indices[len(indices)-2]]:
        indices = np.delete(indices, len(indices)-2)
    else:
        indices = np.delete(indices, len(indices)-1)                
                        
print(indices2)

if len(indices)>len(indices2):
    for i in range(0, len(indices2)-1):
        if indices[i+1]<indices2[i+1]:
            if time_series[indices[i+1]]>time_series[indices[i]]: 
                indices = np.delete(indices, i)
            else:
                indices = np.delete(indices, i+1)   
print('peaks', len(indices))
print('valleys', len(indices2))  


if len(indices2)>len(indices):
    if indices2[0]<indices[0]:
        i=0
        while(i<=len(indices)-2):
            if len(indices)<=len(indices2):
                if indices2[i+1]<indices[i]:
                    if time_series[indices2[i]]<time_series[indices2[i+1]]:
                        indices2 = np.delete(indices2, i+1)
                    else:
                        indices2 = np.delete(indices2, i)
                    i = i -1
            i=i+1
        

print(np.max(time_series[indices2]))
print(indices2)
    
for i in range(0, len(indices2)-1):
    print(i)
    if time_series[indices2[i]]>b:
        indices2 = np.delete(indices2, i)

if len(indices) == len(indices2):
    for i in range(0, len(indices)-2):
        if indices[i+1]<indices2[i+1]:
            if time_series[indices[i]]>time_series[indices[i+1]]:
                indices = np.delete(indices, i+1)
            else:
                indices = np.delete(indices, i)

            
            
          
while(len(indices2)<len(indices)):
    if time_series[indices[len(indices2)-1]]>time_series[indices[len(indices2)]]:
        indices = np.delete(indices, len(indices2))
    else:
        indices = np.delete(indices, len(indices2)-1)

if len(indices2)>len(indices):
    i=0
    while(i<=len(indices)-2):
        if indices2[i+1]<indices[i]:
            if time_series[indices2[i]]<time_series[indices2[i+1]]:
                indices2 = np.delete(indices2, i+1)
            else:
                indices2 = np.delete(indices2, i)
            i = i -1
        i=i+1


for i in range(0, len(indices)-2):
    if indices2[i+1]<indices[i]:
        if time_series[indices2[i]]<time_series[indices2[i+1]]:
            indices2 = np.delete(indices2, i+1)
        else:
            indices2 = np.delete(indices2, i) 
print(indices)
print(indices2)

if len(indices) >= len(indices2):
   for i in range(0, len(indices2)-1):    
       if indices[i+1]<indices2[i+1]:
           if time_series[indices[i]]>time_series[indices[i+1]]:
               indices = np.delete(indices, i+1)
           else:
               indices = np.delete(indices, i)
                
                



if len(indices2)>len(indices):
    if indices2[len(indices2)-1]>indices[len(indices)-1]:
        if indices2[0]<indices[0]:
            indices2 = np.delete(indices2, len(indices2)-1)
    


if len(indices2)<=len(indices):
    for i in range(0, len(indices2)-1):
        if (indices2[i]<indices[i])&(indices2[i+1]<indices[i]):
            print(i)

       
for i in range(0, len(indices)):
    print(i, time_series[indices[i]])
print(indices)

        


index = []
if len(indices2)>len(indices):
    for i in range(0, len(indices)):
        if indices[i]>indices2[len(indices2)-1]:
            index.append(i)
print(index)
for i in reversed(index):
    if time_series[indices[i]]!=np.max(time_series[indices[index]]):
        indices = np.delete(indices, i)

for i in range(0, len(indices2)):
    print(i, time_series[indices2[i]])
print(indices2)

for i in range(0, len(indices)):
    print(i, time_series[indices[i]])
print(indices)

i=1
while(i<=len(indices2)-2):
    if indices[i-1]<indices2[i]:
        if indices[i]>indices2[i+1]:
            print(i)
            print(i+1)
            if time_series[indices2[i]]<time_series[indices2[i+1]]:
                indices2 = np.delete(indices2, i+1)
            else:
                indices2 = np.delete(indices2, i)
    i=i+1   
a=[]
j=0
e=[]
f=[]
for j in range(0, len(indices2)-2):
    print('j=',j)
    for i in range(0, len(indices)):
        if (indices[i]>indices2[j])&(indices[i]<indices2[j+1]):
            a.append(i)
            print(indices[i])
            print(indices2[j], indices2[j+1])
    print('len(a)',len(a))
    if len(a)>1:
        e = a
        print(e, time_series[indices[e]])
        b=np.max(time_series[indices[e]])
        print(b)
        for el in e:
            if time_series[indices[el]]!=b:
                print(el)
                f.append(el)
    a=[]
for fi in reversed(f):
    indices = np.delete(indices, fi)
 
a=[]
j=0
e=[]
f=[]
for j in range(0, len(indices)-1):
    print('j=',j)
    for i in range(0, len(indices2)-1):
        if (indices2[i+1]>indices[j])&(indices2[i+1]<indices[j+1]):
            a.append(i+1)
            print(indices[j], indices[j+1])
            print(indices2[i+1])
    print('len(a)',len(a))
    if len(a)>1:
        e = a
        print(e, time_series[indices2[e]])
        b=np.min(time_series[indices2[e]])
        print(b)
        for el in e:
            if time_series[indices2[el]]!=b:
                print(el)
                f.append(el)
    a=[]

for fi in reversed(f):
    indices2 = np.delete(indices2, fi)

             
if len(indices) == len(indices2):
    print('Good') 
else: 
    print('False')
plt.figure(12)
plt.plot(time_series)
plt.plot(indices, time_series[indices], 'gx')
plt.plot(indices2, time_series[indices2], 'rv')
plt.show()    
#EXTRACT FEATURES
if len(indices) == len(indices2):
    AA = np.mean(0.5*(time_series[indices]-time_series[indices2]))
    AD = np.mean(0.5*(time_series[indices]+time_series[indices2]))
    print('AVERAGE OF AMPLITUDE', AA)
    print('AVERAGE MEDIUM DISTANCE', AD)

    MP = np.max(time_series[indices])
    MV = np.min(time_series[indices2])
    print('MAXIMUM OF PEAK', MP)
    print('MINIMUM OF VALLEY', MV)
    print('NUMBER OF PEAKS', len(indices))

#PLOT VALLEYS
plt.figure(13)
plt.plot(time_series)
plt.plot(indices2, time_series[indices2], "rv")
plt.show()
#PLOT PEAKS AND VALLEYS
plt.figure(14)
plt.plot(time_series)
plt.plot(indices, time_series[indices], "gx")
plt.plot(indices2, time_series[indices2], "rv")
plt.show()                  
 
plt.figure(15)
plt.axes(projection='polar')

# Set the title of the polar plot
plt.title('Circle in polar format:r=R')

# Plot a circle with radius 2 using polar form
rads = np.arange(0, (2*np.pi), 0.01)

for radian in rads:
    plt.polar(radian,AA*np.sin(len(indices)*radian)+AD,'g.') 

plt.polar(t, r, 'b')
# Display the Polar plot
plt.show()

plt.figure(16)
rads = np.linspace(0, (2*np.pi), num = len(time_series))
new = []
for radian in rads:
    new.append(AA*np.sin(len(indices)*radian-np.pi/2)+AD)
plt.plot(new,'g')
plt.show()

plt.figure(17)
plt.plot(new, 'g')
plt.plot(time_series,'b')
interactive(False)
plt.show()




       

cv2.waitKey(0)
cv2.destroyAllWindows()
