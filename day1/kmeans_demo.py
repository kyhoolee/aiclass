import numpy as np
from sklearn.datasets import load_iris
import sklearn.cluster as cluster

from sklearn.manifold import TSNE
from matplotlib import pyplot

data = load_iris()
X = data.data
tsne = TSNE(2).fit_transform(X)
Y = data.target

kmeans = cluster.KMeans(n_clusters=3).fit(X)

# pyplot.scatter(x=tsne[:, 0], y=tsne[:, 1], c=kmeans.labels_ / 2.0)
# pyplot.figure()
# pyplot.scatter(x=tsne[:, 0], y=tsne[:, 1], c=Y / 2.0)
# pyplot.show()


def my_kmeans(x, k):
    def relabel(point, centroids):
        return np.argmin([np.linalg.norm(c - point) for c in centroids])
    centroids = np.array([x[i] for i in np.random.choice(len(x), k, replace=False)])
    labels = np.zeros(len(x))
    while True:
        # pyplot.scatter(x=x[:, 0], y=x[:, 1], c=labels / 2.0)
        # pyplot.scatter(x=centroids[:, 0], y=centroids[:, 1], marker="D")
        # pyplot.show()
        new_labels = np.array([relabel(point, centroids) for point in x])
        if (new_labels == labels).all():
            return centroids, labels
        labels = new_labels
        centroids = np.array([np.mean([x[i] for i in range(len(x)) if labels[i] == j], axis=0)
                              for j in range(k)])

centroids, labels = my_kmeans(X[:, :2], 3)

pyplot.scatter(x=X[:, 0], y=X[:, 1], c=kmeans.labels_ / 2.0)
pyplot.figure()
pyplot.scatter(x=X[:, 0], y=X[:, 1], c=labels / 2.0)
pyplot.scatter(x=centroids[:, 0], y = centroids[:, 1], marker="D")
pyplot.show()