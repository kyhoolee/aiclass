import numpy as np
import gensim
# from dataset import BowFileDataset as Dataset
from dataset import TfidfFileDataset as Dataset


if __name__ == "__main__":
    filename = "./data/vne.txt"
    data = Dataset(stopword_file="./data/stopwords.txt")
    data.load_vocab(filename, 10000)
    bow_corpus, text = data.load(filename)
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=20, id2word=data.index_to_word, passes=5, workers=2)

    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

    print("__________________________________")
    for i in np.random.choice(len(text), 5):
        print(text[i])
        for topic, wt in lda_model[bow_corpus[i]]:
            print(wt, "x", lda_model.print_topic(topic))
        print("_______________")
