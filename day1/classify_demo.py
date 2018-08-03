import numpy as np
from sklearn.datasets import load_iris, fetch_mldata
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

data = load_iris()
#data = fetch_mldata("MNIST original", data_home="./data")
X = data.data
Y = data.target

tsne = TSNE(2).fit_transform(X)
plt.scatter(x=tsne[:, 0], y=tsne[:, 1], c=Y)
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

for classifier in (LogisticRegression(), LinearSVC(), GradientBoostingClassifier(), RandomForestClassifier()):
    classifier.fit(X_train, Y_train)
    Y_result = classifier.predict(X_test)
    print(type(classifier))
    print(classification_report(y_true=Y_test, y_pred=Y_result))

