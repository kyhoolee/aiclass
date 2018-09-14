import pandas as pd
import gensim
# from dataset import BowFileDataset as Dataset
from dataset import TfidfFileDataset as Dataset


if __name__ == "__main__":
    filename = "./data/vne.txt"
    data = Dataset(stopword_file="./data/stopwords.txt")
    data.load_vocab(filename, 10000)
    bow_corpus = data.load(filename)
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=20, id2word=data.index_to_word, passes=5, workers=2)

    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))


