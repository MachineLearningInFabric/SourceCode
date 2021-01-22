# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 20:14:53 2019

@author: 14103
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 19:55:09 2019

@author: 14103
"""


#IMPORT LIBRARY
import pandas as pd #USING PANDAS TO READ DATASET
import matplotlib.pyplot as plt #USING MATPLOTLIB TO VISUALIZE DATA
from sklearn.cluster import KMeans #USING KMEANS TO CLUSTER DATA
from matplotlib import interactive #USING INTERACTIVE OF MATPLOTLIB TO PLOT MULTIDISPLAY FIGURE
import numpy as np #USING NUMPY TO WORK WITH ARRAY
from sklearn.decomposition import PCA #USING PCA TO REDUCE DIMENSION OF DATA


#IMPORT DATA
#USING PANDA TO READ DATA IN EXCEL FILE '111data.xls' AND SKIP ROW 8&9 BECAUSE THERE ARE NO FIGURE IN ROW 8 AND 9
dataset = pd.read_excel('111data.xls', sheet_name='Cotation', skiprows=[8, 9]) 
dataset.describe() #EXPLORE DATASET
#IMPORT COLUMNS OF NECESSARY INFORMATION AND ABANDON THE COLUMN OF DRAPE's PICTURE
df1 = dataset[["N°C1235","AA\naverage amplitude","AD\naverage distance","MP\nmaximum peak", "MV\nminimum valley", "No.P\nnumber of peaks",
               "Weight in g/m²", "Nom commercial ou coloris", "Composition", "Epaisseur en mm", "Armure", "Contexture Chaîne / Trame",
               "Bending Chaine", "Bending Trame", "Drape Coefficient", "Nb plis", "CisT","CisC", "FlexT", "FlexC", "Coloris", "Motifs",
               "First cluster", "Second Cluster", "Group"]]
#THE FEATURE N: Number of fabric 
# AA\naverage amplitude :  average amplitude of drape 
# AD\naverage distance : average distance from zero to medium of sine wave
# MP\nmaximum peak : the maximum amplitude of drape
# MV\nminimum valley : the minimum amplitude of drape
# No.P\nnumber of peaks : number of peaks = number of valleys = number of node in drape
# Weight in g/m² : weight of fabric per 1 m²
# The rest are others mechanical properties of fabric

df1.head()
X1 = df1.iloc[:, [1, 2, 3, 4]].values #FOR CLUSTER ACCORDING TO DRAPE FEATURES
X2 = df1.iloc[:, [5]].values #FOR CLUSTER KMEANS ACCORDING TO NUMBER OF PEAKS
X3 = df1.iloc[:, [6]].values #FOR CLUSTER ACCODING TO WEIGHTS 
X4 = df1.iloc[:, [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]].values #FOR SHOWING THE NEAREST POINT Ò MECHANICAL PROPERTIES
X5 = df1.iloc[:, [1, 2, 3, 4, 5, 6]].values #FOR KNN
X6 = df1.iloc[:, [22]] #FIRST CLUSTER
X7 = df1.iloc[:, [23]]  #SECOND CLUSTER
X8 = df1.iloc[:, [1, 2, 3, 4]] #FOUR DIMENSIONS
X9 = df1.iloc[:, [24]] #GROUP
X10 = df1.iloc[:, [1,2,3,4,5,6]]
label = df1.iloc[:, [0]].values #FOR KNN LABEL Y

#STANDARD SCALER DATA
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X1)
scaler = StandardScaler().fit(X1)

#FIND OPTIMAL K FOR KMEANS CLUSTER BY ELBOW METHOD
Sum_of_squared_distances = []
K = range(1,20)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)

plt.figure(1)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
interactive(True)
plt.show()

#FROM FIGURE 1: THE ELBOW POINT IS IN K=6, SO CHOOSE K = 6

#KMEANS CLUSTER DATASET OF DRAPE
reduced_data = PCA(n_components=2).fit_transform(X)
pca = PCA(n_components=2).fit(X)

#CLUSTER DATASET IN X1 WITH K = 6
kmeans = KMeans(init='random', n_clusters=6, n_init=10, max_iter=300, 
    tol=1e-04, random_state=0)
kmeans.fit(reduced_data)
centers = kmeans.cluster_centers_

#VISUALIZE KMEANS CLUSTER ON DATASET
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(2)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the drape dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
interactive(False)
plt.show()


#TAKING VALUES FOR EACH CLUSTER
def ClusterIndicesNumpy(clustNum, labels_array): #numpy 
    return np.where(labels_array == clustNum)[0]

#CLUSTER OF KMEANS
# nc is an array including all different elements in kmeans.labels_
nc = np.unique(kmeans.labels_)
a = np.array([])
A = []
B = []

#THE FIRST ITERATION TO TAKE ALL POSITION VALUES IN EACH CLUSTER OF DATASET X1 TO ARRAY B
for i in range(0, len(nc)):
    A = ClusterIndicesNumpy(i, kmeans.labels_)
    B.append(A)
    A = []

C = []
D = []
#THE SECOND ITERATION : EACH CLUSTER IN  B WILL BE CLUSTERED AGAIN BY KMEANS METHOD BASED ON THE NUMBER OF NODE AND SAVED IN D
for i in range(0, len(B)):
    C = KMeans(init='random', n_clusters=len(np.unique(X2[B[i]])), n_init=10, max_iter=300, 
    tol=1e-04, random_state=0)
    C.fit(X2[B[i]])  
    D.append(C)
    C = []
  

#PREDICT MECHANICAL PROPERTIES OF A NEW OBSERVATION 
new = [[13.12,	141.93,	169.38,	113.77,	8,	111.1538]]
#TAKING DRAPE's SHAPE DATA TO CLUSTER
new1 = np.array(new[0][0:4]).reshape((1,4))
#PREPROCESSING DATA
xx = scaler.transform(new1)
yy = pca.transform(xx)
#PREDICT CLUSTER OF DRAPE SHAPE DATA
zz = kmeans.predict(yy)
#DATA IN CLUSTER OF ZZ
Datazz = X2[B[int(float(zz))]]
#PREDICT CLUSTER OF NUMBER OF NODE DATA IN GROUP B[int(float(zz))] 
a = D[int(float(zz))].predict([[new[0][4]]])
#POSITION OF ALL ELEMENTS IN THIS CLUSTER
b = ClusterIndicesNumpy(a, D[int(float(zz))].labels_)
number = X2[B[int(float(zz))]][b]
#DATA OF ALL FABRIC IN CLUSTER b IN TO USE KNN
datasetp =  X5[B[int(float(zz))]][b]
y = label[B[int(float(zz))]][b].ravel()
#USING KNN FOR DATA IN THIS CLUSTER
from sklearn.neighbors import KNeighborsClassifier
nbrs = KNeighborsClassifier(n_neighbors=1).fit(datasetp,y)
#PREDICT THE NEAREST POINT IN THIS CLUSTER TO A NEW OBSERVATION
predictt = nbrs.predict(np.array(new))
print('Number of the nearest fabric : ', predictt)
#PRINT MECHANICAL PROPERTIES OF THE NEAREST POINT TO A NEW OBSERVATION
for i in range(0, len(label)):
    if int(float(predictt)) == label[i]:
        mc = X4[i]
print('The mechanical properties of nearest point is : ')
for i in range(0, len(mc)):
    print(mc[i])

for i in range(0, len(B)):
    for j in range(0, len(np.unique(X2[B[i]]))):
        b = ClusterIndicesNumpy(j, D[int(float(i))].labels_)
        print('First Cluster Group :',i,' Second Cluster Group',j, 'Name of Fabric in this group: ',label[B[int(float(i))]][b].ravel())
DataDescribe = df1.iloc[:, [1, 2, 3, 4, 5, 6, 14]].describe()
print(DataDescribe)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

clf = RandomForestClassifier(n_estimators = 50, max_depth = 4)
X = X10
y = dataset.Group
scores = []
num_features = len(X.columns)
for i in range(num_features):
    col = X.columns[i]
    score = np.mean(cross_val_score(clf, X[col].values.reshape(-1,1), y, cv=10))
    scores.append((int(score*100), col))

print(sorted(scores, reverse = True))











