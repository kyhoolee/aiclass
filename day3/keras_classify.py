import codecs
import time

import numpy as np

import dataset
from trainer import Model

from keras.layers import Dropout, Dense, Conv1D, MaxPooling1D, Flatten, Embedding, LSTM
from keras.models import Sequential


def gen_cnn(input_dim, input_length, class_count):
    embedding_dim = 50
    hidden_dims = [50, 50]

    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=input_length, init="uniform"))
    # model.add(Dropout(0.1, input_shape=(input_length, embedding_dim)))
    model.add(Conv1D(nb_filter=200, filter_length=4))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Dropout(0.1))
    model.add(Conv1D(nb_filter=100, filter_length=3))
    model.add(MaxPooling1D(pool_length=3))
    model.add(Flatten())
    for hidden_dim in hidden_dims:
        model.add(Dense(hidden_dim, activation="relu"))
        model.add(Dropout(0.1))
    model.add(Dense(class_count, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    return model


def gen_lstm(input_dim, input_length, class_count):
    embedding_dim = 50
    lstm_dim = 200
    hidden_dims = [50, 50]

    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=input_length, init="uniform"))
    # model.add(Dropout(0.1, input_shape=(input_length, embedding_dim)))
    model.add(LSTM(lstm_dim))
    for hidden_dim in hidden_dims:
        model.add(Dense(hidden_dim, activation="relu"))
        model.add(Dropout(0.1))
    model.add(Dense(class_count, activation="softmax"))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
    return model


def run():
    train_file = "train_sentiment.txt"
    test_file = "test_sentiment.txt"
    vocab_file = "sentiment.vocab"

    train_set = dataset.TextFileDataset()
    train_set.load_vocab(train_file, 5000)
    train_set.save_vocab(vocab_file)
    train_set.load(train_file)
    test_set = dataset.TextFileDataset()
    test_set.reload_vocab(vocab_file)
    test_set.load(test_file)

    trainer = Model()
    trainer.max_epoch = 10
    trainer.model = gen_lstm(train_set.vocab_size(), train_set.input_length, train_set.num_classes())
    trainer.train(train_set, test_set, batch_size=32, verbose=True)


if __name__ == "__main__":
    run()
