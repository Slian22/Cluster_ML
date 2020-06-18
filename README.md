`Cluster_ML`

# 任务清单

## 1、数据预处理

### 1.1 数据处理缺失值

无缺失值处理，因此无函数实现。

### 1.2 数据转化

```python
def Replace_Data(inputfile):
    return Data
#将数据里面的Month类型和VisitorType类型还有Weekend类型进行数值转化；
#返回处理完的数据DataFrame类型
```

- [x] 已实现

### 1.3 数据相关性可视化*

```python
def Corr(inputfile):
    return True
#输出相关性，相关性热力图，分层相关性热力图
```

- [x] 已实现

## 2、聚类

### 2.1 K-MEANS聚类

Parameters

- **n_clusters***int, default=8*

  The number of clusters to form as well as the number of centroids to generate.

- **init***{‘k-means++’, ‘random’, ndarray, callable}, default=’k-means++’*

  Method for initialization:‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. See section Notes in k_init for more details.‘random’: choose `n_clusters` observations (rows) at random from data for the initial centroids.If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.If a callable is passed, it should take arguments X, n_clusters and a random state and return an initialization.

- **n_init***int, default=10*

  Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.

- **max_iter***int, default=300*

  Maximum number of iterations of the k-means algorithm for a single run.

- **tol***float, default=1e-4*

  Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence. It’s not advised to set `tol=0` since convergence might never be declared due to rounding errors. Use a very small number instead.

- **precompute_distances***{‘auto’, True, False}, default=’auto’*

  Precompute distances (faster but takes more memory).‘auto’ : do not precompute distances if n_samples * n_clusters > 12 million. This corresponds to about 100MB overhead per job using double precision.True : always precompute distances.False : never precompute distances.*Deprecated since version 0.23:* ‘precompute_distances’ was deprecated in version 0.22 and will be removed in 0.25. It has no effect.

- **verbose***int, default=0*

  Verbosity mode.

- **random_state***int, RandomState instance, default=None*

  Determines random number generation for centroid initialization. Use an int to make the randomness deterministic. See [Glossary](https://scikit-learn.org/stable/glossary.html#term-random-state).

- **copy_x***bool, default=True*

  When pre-computing distances it is more numerically accurate to center the data first. If copy_x is True (default), then the original data is not modified. If False, the original data is modified, and put back before the function returns, but small numerical differences may be introduced by subtracting and then adding the data mean. Note that if the original data is not C-contiguous, a copy will be made even if copy_x is False. If the original data is sparse, but not in CSR format, a copy will be made even if copy_x is False.

- **n_jobs***int, default=None*

  The number of OpenMP threads to use for the computation. Parallelism is sample-wise on the main cython loop which assigns each sample to its closest center.`None` or `-1` means using all processors.*Deprecated since version 0.23:* `n_jobs` was deprecated in version 0.23 and will be removed in 0.25.

- **algorithm***{“auto”, “full”, “elkan”}, default=”auto”*

  K-means algorithm to use. The classical EM-style algorithm is “full”. The “elkan” variation is more efficient on data with well-defined clusters, by using the triangle inequality. However it’s more memory intensive due to the allocation of an extra array of shape (n_samples, n_clusters).For now “auto” (kept for backward compatibiliy) chooses “elkan” but it might change in the future for a better heuristic.*Changed in version 0.18:* Added Elkan algorithm

Attributes

- **cluster_centers_***ndarray of shape (n_clusters, n_features)*

  Coordinates of cluster centers. If the algorithm stops before fully converging (see `tol` and `max_iter`), these will not be consistent with `labels_`.

- **labels_***ndarray of shape (n_samples,)*

  Labels of each point

- **inertia_***float*

  Sum of squared distances of samples to their closest cluster center.

- **n_iter_***int*

  Number of iterations run.

```python
def K_Means_(inputfile,n):
    return True
#n为想聚到多少类
#用肘部法确定最佳的K值
#输出从1-n每个聚类的效果和图片
```

```powershell
1.K-means算法在多种不同情况下的聚类表现得并不太好，我们可以看到K-means得到的簇更偏向于球形，这 意味着K-means算法不能处理非球形簇的聚类问题，而现实中数据的分布情况是十分复杂的，所以K-means 算法不太适用于现实大多数情况。 

2.K-means算法需要预先确定聚类的个数。

3.K-means算法对初始选取的聚类中心点敏感。我们可以看到在Aggression数据聚类中，K-means算法会把不 同簇聚合在一起来满足球形状簇的聚类，而这种情况下得到的中心点在两个簇中间。这说明此时K-means陷 入了一个局部最优解，而陷入局部最优的一个原因是初始化中心点的位置不太好。
```

- [x] 已实现

### 2.2 AgglomerativeClustering 聚类

Parameters

- **n_clusters***int or None, default=2*

  The number of clusters to find. It must be `None` if `distance_threshold` is not `None`.

- **affinity***str or callable, default=’euclidean’*

  Metric used to compute the linkage. Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”. If linkage is “ward”, only “euclidean” is accepted. If “precomputed”, a distance matrix (instead of a similarity matrix) is needed as input for the fit method.

- **memory***str or object with the joblib.Memory interface, default=None*

  Used to cache the output of the computation of the tree. By default, no caching is done. If a string is given, it is the path to the caching directory.

- **connectivity***array-like or callable, default=None*

  Connectivity matrix. Defines for each sample the neighboring samples following a given structure of the data. This can be a connectivity matrix itself or a callable that transforms the data into a connectivity matrix, such as derived from kneighbors_graph. Default is None, i.e, the hierarchical clustering algorithm is unstructured.

- **compute_full_tree***‘auto’ or bool, default=’auto’*

  Stop early the construction of the tree at n_clusters. This is useful to decrease computation time if the number of clusters is not small compared to the number of samples. This option is useful only when specifying a connectivity matrix. Note also that when varying the number of clusters and using caching, it may be advantageous to compute the full tree. It must be `True` if `distance_threshold` is not `None`. By default `compute_full_tree` is “auto”, which is equivalent to `True` when `distance_threshold` is not `None` or that `n_clusters` is inferior to the maximum between 100 or `0.02 * n_samples`. Otherwise, “auto” is equivalent to `False`.

- **linkage***{“ward”, “complete”, “average”, “single”}, default=”ward”*

  Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion.ward minimizes the variance of the clusters being merged.average uses the average of the distances of each observation of the two sets.complete or maximum linkage uses the maximum distances between all observations of the two sets.single uses the minimum of the distances between all observations of the two sets.*New in version 0.20:* Added the ‘single’ option

- **distance_threshold***float, default=None*

  The linkage distance threshold above which, clusters will not be merged. If not `None`, `n_clusters` must be `None` and `compute_full_tree` must be `True`.*New in version 0.21.*

Attributes

- **n_clusters_***int*

  The number of clusters found by the algorithm. If `distance_threshold=None`, it will be equal to the given `n_clusters`.

- **labels_***ndarray of shape (n_samples)*

  cluster labels for each point

- **n_leaves_***int*

  Number of leaves in the hierarchical tree.

- **n_connected_components_***int*

  The estimated number of connected components in the graph.*New in version 0.21:* `n_connected_components_` was added to replace `n_components_`.

- **children_***array-like of shape (n_samples-1, 2)*

  The children of each non-leaf node. Values less than `n_samples` correspond to leaves of the tree which are the original samples. A node `i` greater than or equal to `n_samples` is a non-leaf node and has children `children_[i - n_samples]`. Alternatively at the i-th iteration, children[i][0] and children[i][1] are merged to form node `n_samples + i`

```python
def Agglo(inputfile,n):
    return True
#输入文件不带最后一列
#输出聚类以后的图形
```



### 2.3 DBSCAN聚类

Parameters

- **X***{array-like, sparse (CSR) matrix} of shape (n_samples, n_features) or (n_samples, n_samples)*

  A feature array, or array of distances between samples if `metric='precomputed'`.

- **eps***float, default=0.5*

  The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster. This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.

- **min_samples***int, default=5*

  The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.

- **metric***string, or callable*

  The metric to use when calculating distance between instances in a feature array. If metric is a string or callable, it must be one of the options allowed by [`sklearn.metrics.pairwise_distances`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances) for its metric parameter. If metric is “precomputed”, X is assumed to be a distance matrix and must be square during fit. X may be a [sparse graph](https://scikit-learn.org/stable/glossary.html#term-sparse-graph), in which case only “nonzero” elements may be considered neighbors.

- **metric_params***dict, default=None*

  Additional keyword arguments for the metric function.*New in version 0.19.*

- **algorithm***{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’*

  The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors. See NearestNeighbors module documentation for details.

- **leaf_size***int, default=30*

  Leaf size passed to BallTree or cKDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.

- **p***float, default=2*

  The power of the Minkowski metric to be used to calculate distance between points.

- **sample_weight***array-like of shape (n_samples,), default=None*

  Weight of each sample, such that a sample with a weight of at least `min_samples` is by itself a core sample; a sample with negative weight may inhibit its eps-neighbor from being core. Note that weights are absolute, and default to 1.

- **n_jobs***int, default=None*

  The number of parallel jobs to run for neighbors search. `None` means 1 unless in a [`joblib.parallel_backend`](https://joblib.readthedocs.io/en/latest/parallel.html#joblib.parallel_backend) context. `-1` means using all processors. See [Glossary](https://scikit-learn.org/stable/glossary.html#term-n-jobs) for more details. If precomputed distance are used, parallel execution is not available and thus n_jobs will have no effect.

Returns

- **core_samples***ndarray of shape (n_core_samples,)*

  Indices of core samples.

- **labels***ndarray of shape (n_samples,)*

  Cluster labels for each point. Noisy samples are given the label -1.

```python
def DBS_(inputfile):
    return True
#传入文件需带最后一列
#输出下列内容
#Estimated number of clusters: 6
#Estimated number of noise points: 12225
#Homogeneity: 0.003
#Completeness: 0.023
#V-measure: 0.006
#Adjusted Rand Index: -0.013
#Adjusted Mutual Information: 0.005
#Silhouette Coefficient: -0.325
#聚类图形
```



- [x] 已实现

## 3、聚类评估

### 3.1 聚类->无监督评估

```python
def Judge_1(inputfile,n):
    return True
#n为聚成几类
#输出KMeans、AgglomerativeClustering、DBSCAN聚类效果
```

- [x] 已实现

### 3.2 聚类->有监督评估

```python
def Judge_2(inputfile,n):
    return True
#同上
```

- [x] 已实现

# 项目实现

```python
#%%

import pandas as pd
import numpy as np
data=pd.read_csv('data/online_shoppers_intention.csv')
data.head()

#%%

from Quick_Init import *
data=Replace_Data('data/online_shoppers_intention_1.csv')
data.head()
data.to_csv('data/trans_1.csv')

#%%

data.describe().round(2).T
#数据描述

#%%

from Quick_Init import *
K_Means_('data/trans_1.csv',10)

#%%

from Quick_Init import *
Judge_1('data/trans_1.csv',2)

#%%

from Quick_Init import *
Judge_2('data/trans_2.csv',5)

#%%

from Quick_Init import *
DBS_('data/trans_2.csv')


#%%

from Quick_Init import *
Agglo('data/trans_1.csv',5)
```