import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
''''
def TSNE_(data_zs):
    from sklearn.manifold import TSNE
    tsne = TSNE()
    tsne.fit_transform(data_zs)  # 进行数据降维
    tsne = pd.DataFrame(tsne.embedding_, index=data_zs.index)  # 转换数据格式
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    d = tsne[data_zs[u'聚类类别'] == 0]
    plt.plot(d[0], d[1], 'r.')
    d = tsne[data_zs[u'聚类类别'] == 1]
    plt.plot(d[0], d[1], 'go')
    d = tsne[data_zs[u'聚类类别'] == 2]
    plt.plot(d[0], d[1], 'b*')
    plt.show()
'''
def draw_silhouette_score(inputfile,n):
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    X = pd.read_csv(inputfile)
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)
    Scores = []  # 存放轮廓系数
    for k in range(2,n):
        estimator = KMeans(n_clusters=k)  # 构造聚类器
        estimator.fit(X)
        Scores.append(silhouette_score(X, estimator.labels_, metric='euclidean'))
    xxx = range(2,n)
    plt.xlabel('k')
    plt.ylabel('Silhouette Coefficient')
    plt.plot(xxx, Scores, 'o-')
    plt.show()
    return True
def draw_(inputfile,k):
    from sklearn.cluster import KMeans
    X=pd.read_csv(inputfile)
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=k)
    X1 = np.array(X)
    kmeans.fit(X1)
    print(kmeans.labels_)  # 类别
    print(kmeans.cluster_centers_)  # 聚类中心
    markers = [['*', 'b'], ['o', 'r'],['.','c'],[',','g'],['^','y'],['<','m'],['>','g'],['s','b'],['h','r'],['+','c'],['d','r']]
    for i in range(k):
        members = kmeans.labels_ == i  # members是布尔数组
        plt.scatter(X1[members, 0], X1[members, 1], s=60, marker=markers[i][0], c=markers[i][1], alpha=0.5)
    plt.title('KmeansClustering')
    plt.show()
def Corr(inputfile):
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
    from matplotlib.font_manager import FontProperties
    K = range(1, n)
    X=pd.read_csv(inputfile)
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)
    mean_distortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        r1 = pd.Series(kmeans.labels_).value_counts()  # 统计各个类别的数目
        r2 = pd.DataFrame(kmeans.cluster_centers_)  # 找出聚类中心
        r = pd.concat([r2, r1], axis=1)  # 横向连接，0是纵向
        print(r)
        mean_distortions.append(kmeans.inertia_)
    plt.plot(K, mean_distortions, 'bx-')
    plt.xlabel('k')
    font = FontProperties(size=20)
    plt.ylabel(u'平均畸变程度', fontproperties=font)
    plt.title(u'用肘部法确定最佳的K值', fontproperties=font)
    plt.show()
    draw_silhouette_score(inputfile,n)
    return True
def Agglo(inputfile,n):
    import scipy.cluster.hierarchy as sch
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import MinMaxScaler
    # 读取数据
    data = pd.read_csv(inputfile)
    df = MinMaxScaler().fit_transform(data)
    # 建立模型
    model = AgglomerativeClustering(n_clusters=n)
    model.fit(df)
    data['类别标签'] = model.labels_
    print(data.head())
    # 画图
    ss = sch.linkage(df, method='ward')
    sch.dendrogram(ss)
    plt.show()
    # %%
    return True
def DBS_(inputfile):
    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn import metrics
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    # 初始化样本数据
    data = pd.read_csv(inputfile)
    labels_true = data['Revenue']
    X = data.drop('Revenue', axis=1)
    X = StandardScaler().fit_transform(X)
    # 计算DBSCAN
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # 聚类的结果
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels,
                                               average_method='arithmetic'))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))
    # 绘出结果
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
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