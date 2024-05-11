from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

os.chdir(r"D:\Datasets_for_PML_EME")

us = pd.read_csv("USArrests.csv", index_col=0)
# X = us.drop('ID', axis=1)
# y = us['Type']


scaler = StandardScaler()
usscaled = scaler.fit_transform(us)

# Finding the best cluster based on Silhouette
sill = []
for i in np.arange(2,10):
    km = KMeans(n_clusters=i, random_state=2022)
    km.fit(usscaled)
    labels = km.predict(rfmscaled)
    score = silhouette_score(rfmscaled, labels)
    sil.append(score)
    print("K=",i," Score=", score)
    

Ks = np.arange(2,10)
i_max = np.argmax(sil)
best_k = Ks[i_max]
print("Best K =", best_k)

km = KMeans(n_clusters=best_k, random_state=2022)
km.fit(rfmscaled)
labels = km.predict(rfmscaled)

rfm["Cluster"] = labels
rfm.sort_values('Cluster', inplace=True)

# Calculating the centroids
rfm.groupby('Cluster').mean()