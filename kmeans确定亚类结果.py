import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster._hierarchy import linkage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ----------第二问聚类--------

dataz = pd.read_excel('D:/pytorch/2022c建模/汇总信息表.xlsx', sheet_name='高钾')
data = pd.read_excel('D:/pytorch/2022c建模/汇总信息表.xlsx', sheet_name='铅钡含严重风化')


data1 = data.iloc[:, 7:21].values
data2 = dataz.iloc[:, 7:21].values

min_size = min(data1.shape[1], data2.shape[1])

sil_score_q = []
inert_q = []
for i in range(2, min_size):
    kmeans = KMeans(n_clusters=i, n_init=100,max_iter=1000).fit(data1)
    sil_score_q.append(silhouette_score(data1, kmeans.labels_))
    inert_q.append(kmeans.inertia_)
plt.plot(range(2, min_size), sil_score_q, 'o-')
plt.xlabel('k')
plt.show()

plt.plot(range(2, min_size), inert_q, 'o-')
plt.xlabel('k')
plt.show()

sil_score_k = []
inert_k = []
for i in range(2, min_size):
    kmeans = KMeans(n_clusters=i, n_init=100,max_iter=1000).fit(data2)
    sil_score_k.append(silhouette_score(data2, kmeans.labels_))
    inert_k.append(kmeans.inertia_)
plt.plot(range(2, min_size), sil_score_k, 'o-')
plt.xlabel('k')
plt.show()

plt.plot(range(2, min_size), inert_k, 'o-')
plt.xlabel('k')
plt.show()

qinaBa = KMeans(n_clusters=3, n_init=100, max_iter=1000).fit(data1)
gaoK = KMeans(n_clusters=2, n_init=100, max_iter=1000).fit(data2)

pred_index_q = qinaBa.fit_predict(data1)
pred_index_k = gaoK.fit_predict(data2)

print(pred_index_q, pred_index_k)
print(qinaBa.cluster_centers_, gaoK.cluster_centers_)

