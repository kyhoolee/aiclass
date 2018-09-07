import time

import numpy as np
import pandas as pd

from keras.models import model_from_json
from sklearn.metrics import accuracy_score


class Model(object):
    def __init__(self):
        self.model = None
        self.min_epoch = 0
        self.max_epoch = 1
        self.patience = 2
        self.validation_frequency = 1

    def train(self, train, validation, batch_size, class_weight=None, verbose=False):
        saved_model = model_from_json(self.model.to_json())
        best_epoch = None
        timer = time.time()
        if validation is not None:
            best = self.score(validation, batch_size)
        else:
            best = None
        for epoch in np.arange(self.max_epoch):
            loss = []
            for x, y in train.batches(batch_size):
                loss.append(self.model.train_on_batch(x, y, class_weight=class_weight))
            if verbose and (epoch < self.min_epoch or validation is None):
                print("\rEpoch", epoch,
                      "time", time.time() - timer)
                continue
            valid = self.score(validation, batch_size)
            if verbose:
                print("\rEpoch", epoch,
                      "score", valid,
                      "best", best,
                      "time", time.time() - timer)
            if valid > best:
                saved_model.set_weights(self.model.get_weights())
                best_epoch = epoch
                best = valid
            elif self.patience > 0:
                self.patience -= 1
            elif self.patience == 0:
                break
        if validation is not None:
            self.model.set_weights(saved_model.get_weights())
        return best_epoch, best

    def predict_classes(self, x):
        proba = self.predict(x)
        if proba.shape[-1] > 1:
            return proba.argmax(axis=-1)
        else:
            return (proba > 0.5).astype('int32')

    def predict(self, x):
        return self.model.predict(x, verbose=0)

    def evaluate(self, data, batch_size):
        Z = []
        Y = []
        for x, y in data.batches(batch_size):
            Z.append(self.predict_classes(x))
            Y.append(y)
        return np.concatenate(Z), np.concatenate(Y)

    def score(self, data, batch_size):
        Z, Y = self.evaluate(data, batch_size)
        return accuracy_score(y_pred=Z, y_true=Y)
