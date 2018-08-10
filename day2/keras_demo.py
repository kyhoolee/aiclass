import numpy as np
import os
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import classification_report


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


if __name__ == "__main__":
    data = load_iris()
    # data = load_digits()
    X = data.data
    Y = data.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    classifier = Sequential()
    classifier.add(Dense(10, input_dim=X.shape[1], activation="relu"))
    classifier.add(Dense(5, activation="relu"))
    classifier.add(Dense(np.max(Y) + 1, activation="softmax"))
    # optimizer = Adam(lr=0.01, decay=0.01)
    classifier.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    classifier.fit(X_train, Y_train, batch_size=8, epochs=10, verbose=True)
    Y_result = classifier.predict_classes(X_test)
    print(classification_report(y_true=Y_test, y_pred=Y_result))
