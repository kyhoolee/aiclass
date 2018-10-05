from collections import Counter, defaultdict
import numpy as np


def build_dictionary(filename, vocab_size):
    word_counts = Counter()
    doc_freq = defaultdict(int)
    corpus_size = 0
    with open(filename, "r", encoding="utf-8") as source:
        for line in source:
            words = line.strip().split()
            word_counts.update(words)
            for word in set(words):
                doc_freq[word] += 1
            corpus_size += 1
    index = {w[0]: i + 1 for i, w in enumerate(word_counts.most_common(vocab_size))}
    idf = {i: np.log(corpus_size / doc_freq[w]) for w, i in index.items()}
    return index, idf


class BowFileDataset(object):
    def __init__(self, stopword_file):
        with open(stopword_file, "r", encoding="utf-8") as source:
            self.stopwords = set([x.strip() for x in source])

    def load_vocab(self, filename, vocab_size):
        self.word_to_index, self.idf = build_dictionary(filename, vocab_size)
        self.index_to_word = {w: i for i, w in self.word_to_index.items()}

    def process_line(self, line):
        result = Counter()
        result.update([self.word_to_index[w] for w in line
                       if w in self.word_to_index
                       and w not in self.stopwords])
        return list(result.items())

    def load(self, filename):
        result = []
        text = []
        with open(filename, "r", encoding="utf-8") as source:
            for line in source:
                text.append(line)
                words = line.strip().split()
                result.append(self.process_line(words))
        return result, text


class TfidfFileDataset(BowFileDataset):
    def process_line(self, line):
        result = Counter()
        result.update([self.word_to_index[w] for w in line
                       if w in self.word_to_index
                       and w not in self.stopwords])
        return [(i, n / self.idf[i]) for i, n in result.items()]
