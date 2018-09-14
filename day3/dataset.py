import numpy as np
import codecs
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
import pickle


def build_dictionary(filename, vocab_size, delimiter, text_idx, label_idx):
    word_counts = Counter()
    classes = {}
    max_length = 0
    with open(filename, "r", encoding="utf-8") as source:
        for line in source:
            tokens = line.strip().split(delimiter)
            label = tokens[label_idx]
            text = tokens[text_idx]
            words = text.split()
            word_counts.update(words)
            max_length = max(max_length, len(words))
            if label not in classes:
                classes[label] = len(classes)
    return {w[0]: i + 1 for i, w in enumerate(word_counts.most_common(vocab_size))}, classes, max_length


class TextFileDataset(object):
    def __init__(self, delimiter="\t", text_idx=1, label_idx=0):
        self.text_idx = text_idx
        self.label_idx = label_idx
        self.delimiter = delimiter

    def load_vocab(self, filename, vocab_size):
        self.word_to_index, self.classes, self.input_length = build_dictionary(filename,
                                                                               vocab_size,
                                                                               self.delimiter,
                                                                               self.text_idx,
                                                                               self.label_idx)
        self.index_to_word = {w: i for i, w in self.word_to_index.items()}

    def save_vocab(self, filename):
        with open(filename, "wb") as target:
            pickle.dump((self.word_to_index, self.index_to_word, self.classes, self.input_length), target)

    def reload_vocab(self, filename):
        with open(filename, "rb") as source:
            self.word_to_index, self.index_to_word, self.classes, self.input_length = pickle.load(source)

    def vocab_size(self):
        return len(self.word_to_index) + 1

    def num_classes(self):
        return len(self.classes)

    def text(self, line):
        return " ".join(self.index_to_word.get(x, "<UNK>") for x in line)

    def close(self):
        if not self.source_file.closed:
            self.source_file.close()

    def batch(self, batch_size):
        x = []
        y = []
        for i in range(batch_size):
            label, line = next(self.source)
            if line is None:
                break
            x.append(self.process_line(line))
            y.append(self.classes[label])
        return np.array(x), np.array(y)

    def batches(self, batch_size, equal_batches=False):
        alive = True
        while alive:
            result = self.batch(batch_size)
            alive = len(result[0]) == batch_size
            if len(result[0]) == batch_size or (not equal_batches and len(result[0]) > 0):
                yield result
        self.reset()

    def process_line(self, line):
        line = [self.word_to_index.get(w, 0) for w in line]
        if len(line) < self.input_length:
            return line + [0] * (self.input_length - len(line))
        else:
            return line[: self.input_length]

    def get_source(self):
        with open(self.source_file, "r", encoding="utf-8") as source:
            for line in source:
                tokens = line.strip().split(self.delimiter)
                label = tokens[self.label_idx]
                text = tokens[self.text_idx]
                words = text.split()
                yield label, words
            yield None, None

    def reset(self):
        self.source = self.get_source()

    def load(self, filename):
        self.source_file = filename
        self.reset()
