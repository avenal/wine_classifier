#!/usr/bin/python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sklearn.preprocessing as preprocessing

data = pd.read_csv('wine.data')

columns = ['vinyard','alcohol','malic acid','ash','ash alcalinity','magnesium',
           'Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins',
           'Color intensity','Hue','dilute OD280/OD315', 'Proline']
data.columns = columns
vinyard = data['vinyard']
colors = np.array(['red','green','blue'])[np.array([vinyard])-1]

data_processed = preprocessing.MinMaxScaler().fit_transform(data)
model = PCA(n_components=2)
model.fit(data_processed)
data_pca = model.transform(data_processed)
print(data_processed)
np.savetxt(r'wine_normalized.csv',data_processed,delimiter=',')

model = PCA(n_components=2)
model.fit(data_processed)
data_pca = model.transform(data_processed)
plt.scatter(*data_pca.T, s=8, c=colors.reshape(-1))
plt.title('UCI Wine after normalization')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

plt.show()
