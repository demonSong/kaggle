#**sklearn 源码分析系列：neighbors(1)**

## **Nearest Centroid Classifier**
本篇文章主要来实操官方文档中关于【Nearest Neighbors】的相关知识。详见[文档][1]。

>这里分析采用了Ipython notebook.

**加载数据**
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.neighbors.nearest_centroid import NearestCentroid

n_neighbors = 15

# 加载数据
iris = datasets.load_iris()
print(iris)
```

这是sklearn所提供的数据集，后文会分析它们是如何被加载的。此处，我们得到了iris的数据。

**iris数据集分析**
```python
{'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), 'feature_names': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'], 'data': array([[ 5.1,  3.5,  1.4,  0.2],
       [ 4.9,  3. ,  1.4,  0.2],
       [ 4.7,  3.2,  1.3,  0.2],
       [ 4.6,  3.1,  1.5,  0.2])
}
```
控制台输出的部分数据，很简单，输入空间x的特征有四个维度，输出标签分别为0，1，2。

**二维可视化**
由于目前输入样例是四维的特征向量，这里我们挑选两个维度进行可视化。

```python
# 二维可视化
X = iris.data[:,:2]
y = iris.target

cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.figure()

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification")

plt.show()
```

![alt text](http://img.blog.csdn.net/20170313152313653?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDY4ODE0NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


**可视化分类器及数据**
```python
# 可视化分类器及数据
h = .02

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

clf = NearestCentroid()
clf.fit(X, y)

# 计算每个特征向量的最大值和最小值
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# 可视化分类器
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# 可视化数据
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification")

plt.show()
```

![alt text](http://img.blog.csdn.net/20170313152727312?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDY4ODE0NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

背景颜色即为NearestCentroid()分类器。从图中也可以看出，该分类器是把整个空间切分成了三个区域，达到分类的目的。

**模型训练**
```python
# 模型训练
X = iris.data[:,:5]
y = iris.target

clf = NearestCentroid()
clf.fit(X,y)

score = clf.score(X,y)
print(score)

0.926666666667
```
该模型对iris数据集的准确率高达92.67，还是很不错的哟。


[1]:http://scikit-learn.org/stable/modules/neighbors.html
