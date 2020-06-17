#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
def draw_2(inputfile,n):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import AgglomerativeClustering
    from scipy.cluster.hierarchy import dendrogram, linkage
    data = np.random.randint(0, 10, size=[10, 2])
    Z = linkage(data)

    dendrogram(Z)
def draw_(inputfile,k):
    from sklearn.cluster import KMeans
    X=pd.read_csv(inputfile)
    kmeans = KMeans(n_clusters=k)
    X1 = np.array(X)
    kmeans.fit(X1)
    print(kmeans.labels_)  # 类别
    print(kmeans.cluster_centers_)  # 聚类中心
    markers = [['*', 'b'], ['o', 'r']]
    for i in range(2):
        members = kmeans.labels_ == i  # members是布尔数组
        plt.scatter(X1[members, 0], X1[members, 1], s=60, marker=markers[i][0], c=markers[i][1], alpha=0.5)
    plt.title('KmeansClustering')
    plt.show()
def Corr(inputfile):
    import matplotlib.pyplot as plt
    import seaborn as sns
    data=pd.read_csv(inputfile)
    print("相关性：")
    print(data.corr().round(2).T)
    print("相关性热力图：")
    print(sns.heatmap(data.corr()))
    print("分层相关性热力图：")
    print(sns.clustermap(data.corr()))
    return True
def Replace_Data(inputfile):
    data=pd.read_csv(inputfile)
    data['Month'].replace('Jan',1,inplace=True)
    data['Month'].replace('Feb', 2, inplace=True)
    data['Month'].replace('Mar', 3, inplace=True)
    data['Month'].replace('Apr', 4, inplace=True)
    data['Month'].replace('May', 5, inplace=True)
    data['Month'].replace('June', 6, inplace=True)
    data['Month'].replace('Jul', 7, inplace=True)
    data['Month'].replace('Aug', 8, inplace=True)
    data['Month'].replace('Sep', 9, inplace=True)
    data['Month'].replace('Oct', 10, inplace=True)
    data['Month'].replace('Nov', 11, inplace=True)
    data['Month'].replace('Dec', 12, inplace=True)
    data['VisitorType'].replace('Returning_Visitor',1,inplace=True)
    data['VisitorType'].replace('Other', 2, inplace=True)
    data['VisitorType'].replace('New_Visitor', 3, inplace=True)
    data['VisitorType'].replace('New_Visitor', 3, inplace=True)
    data['Weekend'].replace(False, 0, inplace=True)
    data['Weekend'].replace(True, 1, inplace=True)
    return data
def K_Means_(inputfile,n):
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist
    from matplotlib.font_manager import FontProperties
    K = range(1, n)
    X=pd.read_csv(inputfile)
    mean_distortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        mean_distortions.append(
            sum(
                np.min(
                    cdist(X, kmeans.cluster_centers_, metric='euclidean'), axis=1))
            / X.shape[0])
    plt.plot(K, mean_distortions, 'bx-')
    plt.xlabel('k')
    font = FontProperties(size=20)
    plt.ylabel(u'平均畸变程度', fontproperties=font)
    plt.title(u'用肘部法确定最佳的K值', fontproperties=font)
    plt.show()
    draw_(inputfile,k)
    return True
def Agglo(inputfile,n):
    from sklearn.cluster import AgglomerativeClustering
    import scipy.cluster.hierarchy as shc
    data = pd.read_csv(inputfile)
    cluster = AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='ward')
    cluster.fit_predict(data)
    plt.figure(figsize=(10, 7))
    plt.title("Customer Dendograms")
    dend = shc.dendrogram(shc.linkage(data, method='ward'))
    plt.savefig("01.jpg")
    plt.show()
    # %%
    from sklearn.cluster import AgglomerativeClustering
    cluster = AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='ward')
    cluster.fit_predict(data)
    plt.figure(figsize=(10, 7))
    plt.scatter(data[:, 0], data[:, 1], c=cluster.labels_, cmap='rainbow')
    plt.savefig("02.jpg")
    plt.show()
def DBS_(inputfile):
    from sklearn.cluster import DBSCAN
    from sklearn import metrics
    X=pd.read_csv(inputfile)
    db = DBSCAN().fit(X)
    labels = db.labels_
    X['cluster_db'] = labels
    X.sort_values('cluster_db')
    # 查看根据DBSCAN聚类后的分组统计结果（均值）
    print(X.groupby('cluster_db').mean())
    # 我们可以从上面这个图里观察聚类效果的好坏，但是当数据量很大，或者指标很多的时候，观察起来就会非常麻烦。
    # 就是下面这个函数可以计算轮廓系数
    score = metrics.silhouette_score(X, X.cluster_db)
    print(score)
    markers = [['*', 'b'], ['o', 'r']]
    for i in range(2):
        members = db.labels_ == i  # members是布尔数组
        plt.scatter(X[members, 0], X[members, 1], s=60, marker=markers[i][0], c=markers[i][1], alpha=0.5)
    plt.title('DBSCANClustering')
    plt.show()
    return True
def Judge_1(inputfile,n):
    from sklearn.metrics.cluster import silhouette_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    import matplotlib.pyplot as plt
    import mglearn
    X= pd.read_csv(inputfile)
    # 将数据缩放成平均值为 0，方差为 1
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks': (), 'yticks': ()})
    # 需要使用的算法
    algorithms = [KMeans(n_clusters=n), AgglomerativeClustering(n_clusters=n), DBSCAN()]
    # 创建一个随机的簇分配，作为参考
    random_state = np.random.RandomState(seed=0)
    random_clusters = random_state.randint(low=0, high=2, size=len(X))
    axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters, cmap=mglearn.cm3, s=60)
    axes[0].set_title('Random assignment:{:.2f}'.format(silhouette_score(X_scaled, random_clusters)))
    for ax, algorithm in zip(axes[1:], algorithms):
        clusters = algorithm.fit_predict(X_scaled)
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=60)
        ax.set_title('{}:{:.2f}'.format(algorithm.__class__.__name__, silhouette_score(X_scaled, clusters)))
    plt.show()
    return True
def Judge_2(inputfile,n):
    from sklearn.metrics.cluster import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    import numpy as np
    import matplotlib.pyplot as plt
    import mglearn
    data= pd.read_csv(inputfile)
    y = data['Revenue']
    X=data.drop('Revenue', axis=1)
    # 将数据缩放成平均值为 0，方差为 1
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks': (), 'yticks': ()})
    # 需要使用的算法
    algorithms = [KMeans(n_clusters=n), AgglomerativeClustering(n_clusters=n), DBSCAN()]
    # 创建一个随机的簇分配，作为参考
    random_state = np.random.RandomState(seed=0)
    random_clusters = random_state.randint(low=0, high=2, size=len(X))
    axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters, cmap=mglearn.cm3, s=60)
    axes[0].set_title('Random assignment - ARI:{:.2f}'.format(adjusted_rand_score(y, random_clusters)))
    for ax, algorithm in zip(axes[1:], algorithms):
        clusters = algorithm.fit_predict(X_scaled)
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=60)
        ax.set_title('{}- ARI:{:.2f}'.format(algorithm.__class__.__name__, adjusted_rand_score(y, clusters)))
    plt.show()
    return True