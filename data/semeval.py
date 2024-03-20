from bs4 import BeautifulSoup
import glob
from tqdm import tqdm
import spacy

TEXT = "text"
MODEL = "en_core_web_sm"

nlp = spacy.load(MODEL)

def main():
    # read all xml in SemEval dir
    sentences = []
    for filename in glob.glob("data/SemEval/*.xml"):
        with open(filename, 'r') as f:
            data = f.read()
            soup = BeautifulSoup(data, "xml")
            for text in soup.find_all(TEXT):
                sentences.append(text.text)

    for filename in ['data/CitySearch/test.txt', 'data/CitySearch/train.txt']:
        with open(filename, 'r') as f:
            for line in f:
                sentences.append(line.strip())

    sentences_prep = []
    for sentence in sentences:
        sentence  = sentence.replace(",", ".")
        sentence = sentence.replace("n't", " not")
        sentence = ' '.join(sentence.split()).strip()
        sentences_prep.append(sentence)



    conll = []
    sentences_clean = []
    for doc in tqdm(nlp.pipe(sentences_prep, n_process=15), total=len(sentences_prep)):
        idx = 0
        tokens = []
        for token in doc:
            if token.pos_ == "SPACE": continue # ignore space
            conll.append(f"{idx}\t{token.text}\t{token.lemma_}\t{token.pos_}\t{token.tag_}\t_\t{token.head.i}\t{token.dep_}\t_\t_")
            idx += 1
            if token.is_punct or len(token.text) < 2 or token.pos_ in ['DET', 'CONJ', 'AUX'] or token.is_stop or "'" in token.text: continue  # ignore punctuation and single character
            tokens.append(token.text)
        if len(tokens) > 1:
            sentences_clean.append(" ".join(tokens))

    with open("data/restaurant_train.conllu", "w") as f:
        f.write("\n".join(conll))

    with open("data/all_txt_restaurant.txt", "w") as f:
        f.write("\n".join(sentences_clean))

if __name__ == "__main__":
    main()