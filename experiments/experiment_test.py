"""Experiment on the test data."""
import json
import numpy as np

from cat.simple import get_scores, attention, rbf_attention, cosine_variance_attention, mean
from cat.dataset import restaurants_test
from reach import Reach
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict, Counter
from itertools import product

import pandas as pd


GAMMA = {"cbow": 0.04, "sg": 0.04}
BEST_ATT = {"cbow": 30, "sg": 30}
BEST_RBF = {"cbow": 30, "sg": 30}
BEST_COS = {"cbow": 30, "sg": 30}

if __name__ == "__main__":

    scores = defaultdict(dict)
    r = Reach.load("embeddings/restaurant_vecs_w2v_cbow.vec",
                   unk_word="<UNK>")

    d = json.load(open("data/nouns_restaurant.json"))

    nouns = Counter()
    for k, v in d.items():
        if k.lower() in r.items:
            nouns[k.lower()] += v

    embedding_paths = ["embeddings/restaurant_vecs_w2v_cbow.vec", "embeddings/restaurant_vecs_w2v_sg.vec"]
    bundles = ((rbf_attention, attention, cosine_variance_attention, mean), embedding_paths)

    fun2name = {attention: "att", rbf_attention: "rbf", cosine_variance_attention: "cosine", mean: "mean"}


    for att, path in product(*bundles):
        r = Reach.load(path, unk_word="<UNK>")

        model = 'cbow' if 'cbow' in path else 'sg'

        if att == rbf_attention:
            candidates, _ = zip(*nouns.most_common(BEST_RBF[model]))
        if att == cosine_variance_attention:
            candidates, _ = zip(*nouns.most_common(BEST_COS[model]))
        else:
            candidates, _ = zip(*nouns.most_common(BEST_ATT[model]))

        aspects = [[x] for x in candidates]

        for idx, (instances, y, label_set) in enumerate(restaurants_test()):
            print("label_set", label_set)

            s = get_scores(instances,
                           aspects,
                           r,
                           label_set,
                           gamma=GAMMA[model],
                           remove_oov=False,
                           attention_func=att)

            y_pred = s.argmax(1)
            f1_score = precision_recall_fscore_support(y, y_pred)
            f1_macro = precision_recall_fscore_support(y,
                                                       y_pred,
                                                       average="weighted")
            scores[(att, model)][idx] = (f1_score, f1_macro)

    att_score = {k: v for k, v in scores.items() if k[0] == attention}
    att_per_class = [[z[x][0][:-1] for x in range(3)]
                     for z in att_score.values()]
    att_per_class = np.stack(att_per_class).mean(0)
    att_macro = np.mean([v[2][1][:-1] for v in att_score.values()], 0)

    rbf_score = {k: v for k, v in scores.items() if k[0] == rbf_attention}
    rbf_per_class = [[z[x][0][:-1] for x in range(3)]
                     for z in rbf_score.values()]
    rbf_per_class = np.stack(rbf_per_class).mean(0)
    rbf_macro = np.mean([v[2][1][:-1] for v in rbf_score.values()], 0)



    header = ["dataset", "model", "att", "precision", "recall", "f1"]
    row = []
    for k, dataset in scores.items():
        name = fun2name[k[0]]
        for idx, (_, f1_macro) in dataset.items():
            row.append([idx, k[1], name, round(f1_macro[0] * 100, 1), round(f1_macro[1] * 100, 1), round(f1_macro[2] * 100, 1)])

    df = pd.DataFrame(row, columns=header)

    df.to_csv("results_30.csv", index=False)