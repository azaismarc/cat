"""Creating fragments takes a long time so we treat it as a
pre-processing step."""
import logging

from gensim.models import Word2Vec
from cat.fragments import create_noun_counts
from cat.utils import conll2text

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":

    print("Creating noun counts and converting conll to text")

    #paths = ["data/restaurant_train.conllu"]
    #create_noun_counts(paths, "data/nouns_restaurant.json")
    # conll2text(paths, "data/all_txt_restaurant.txt")

    corpus = [x.lower().strip().split()
              for x in open("data/all_txt_restaurant.txt")]
    
    print(f"Corpus size: {len(corpus)}")

    print("Training word2vec model")

    for sg in [0, 1]:

        algo = "cbow" if sg == 0 else "sg"

        print(f"Training {algo} model")

        f = Word2Vec(corpus,
                     sg=sg,
                     negative=5,
                     window=5,
                     ns_exponent=1,
                     vector_size=100,
                     min_count=10,
                     epochs=5,
                     workers=15,
                     seed=42)

        f.wv.save_word2vec_format(f"embeddings/restaurant_vecs_w2v_{algo}.vec")

