from bs4 import BeautifulSoup

RESTAURANT_14_TRAIN = "Restaurants_Train.xml"
RESTAURANT_14_TEST = "Restaurants_Test_Data_phaseB.xml"
RESTAURANT_15_TRAIN = "ABSA-15_Restaurants_Train_Final.xml"
RESTAURANT_15_TEST = "ABSA15_Restaurants_Test.xml"
CITYSEARCH_TEST = "test.txt"
CITYSEARCH_LABEL = "test_label.txt"

D_FILENAME = {
    RESTAURANT_14_TRAIN: {"label": "labels_restaurant_train_2014.txt", "text": "restaurant_train_2014.txt"},
    RESTAURANT_14_TEST: {"label": "labels_restaurant_test_2014.txt", "text": "restaurant_test_2014_tok.txt"},
    RESTAURANT_15_TRAIN: {"label": "labels_restaurant_train_2015.txt", "text": "restaurant_train_2015_tok.txt"},
    RESTAURANT_15_TEST: {"label": "labels_restaurant_test_2015.txt", "text": "restaurant_test_2015_tok.txt"}
}


def main():
    # read all xml in SemEval dir
    for filename in [RESTAURANT_14_TRAIN, RESTAURANT_14_TEST, RESTAURANT_15_TRAIN, RESTAURANT_15_TEST]:
        with open('data/SemEval/'+filename, 'r') as f:
            data = f.read()
            soup = BeautifulSoup(data, "xml")
            labels = []
            sentences = []
            for sentence in soup.find_all("sentence"):
                cat_tag = "aspectCategory" if filename in [RESTAURANT_14_TRAIN, RESTAURANT_14_TEST] else "Opinion"
                categories = sentence.find_all(cat_tag)
                if len(categories) != 1: continue
                labels.append(categories[0]["category"])
                sentences.append(sentence.find("text").text)
    
            with open("data/"+D_FILENAME[filename]["label"], "w") as f:
                f.write("\n".join(labels))

            with open("data/"+D_FILENAME[filename]["text"], "w") as f:
                f.write("\n".join(sentences))

    # same for citysearch
    sentences = []
    with open("data/CitySearch/"+CITYSEARCH_TEST, "r") as f:
        for line in f: sentences.append(line.strip())

    labels = []
    with open("data/CitySearch/"+CITYSEARCH_LABEL, "r") as f:
        for line in f: labels.append(line.strip().split())

    clear_labels = []
    clear_sentences = []
    for label, sentence in zip(labels, sentences):
        if len(label) != 1: continue
        clear_labels.append(label[0])
        clear_sentences.append(sentence)

    with open("data/test_label.txt", "w") as f:
        f.write("\n".join(clear_labels))

    with open("data/test_tok.txt", "w") as f:
        f.write("\n".join(clear_sentences))
          
if __name__ == "__main__":
    main()