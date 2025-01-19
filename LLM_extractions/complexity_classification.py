from sklearn.utils import shuffle
import os, json
import spacy.lang
from util_functions import clean_leaflet
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearn.pipeline import make_pipeline

def prepare_data_with_label_1(filenames):
    sentences = []

    for f in filenames:
        with open(llm_folder + '/' + f, 'r') as file:
            leaflet_llm = json.load(file)

        leaflet_llm = '. '.join(list(map(lambda x : x['excerpt'], leaflet_llm)))

        doc = nlp(leaflet_llm)

        file_sentences = [sent.text.strip(" \n") for sent in doc.sents]
        sentences.extend(file_sentences)
    
    sentences = np.unique(sentences)
    labels = [1 for _ in range(sentences.shape[0])]

    sentences_mean_len = np.mean(list(map(len, sentences)))
    sentences_std_len = np.std(list(map(len, sentences)))

    return sentences, labels, sentences_mean_len, sentences_std_len
        

def prepare_data_with_label_0(filenames, sentences_1, sentences_1_mean_len, sentences_1_std_len):
    sentences = []

    for f in filenames:
        leaflet_no = int(f.split('_')[0])
        leaflet_text = clean_leaflet(leaflet_no)

        doc = nlp(leaflet_text)

        # take only the sentences that weren t already marked as complex by the llm and are shorter or longer than the average sentences_1 length with 2 std dev
        min_len = sentences_1_mean_len - 2 * sentences_1_std_len
        max_len = sentences_1_mean_len + 2 * sentences_1_std_len
        file_sentences = [sent.text.strip(" \n") for sent in doc.sents if sent.text not in sentences_1 and min_len <= len(sent.text) <= max_len]
        sentences.extend(file_sentences)

    sentences = np.unique(sentences)
    labels = [0 for _ in range(sentences.shape[0])]

    return sentences, labels

def prepare_data(filenames):
    sentences_1, labels_1, sentences_1_mean_len, sentences_1_std_len = prepare_data_with_label_1(filenames)
    sentences_0, labels_0 = prepare_data_with_label_0(filenames, sentences_1, sentences_1_mean_len, sentences_1_std_len)
    print('0 label: ', sentences_0.shape)
    print('1 label: ', sentences_1.shape)
    data = np.concatenate((sentences_1, sentences_0), axis=0)
    labels = labels_1 + labels_0

    # shuffle
    data, labels = shuffle(data, labels)

    return data, labels

# def translate_to_pos(data):


def train_tf_idf_vectorizer(vocabulary, data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3, random_state = 0)

    print('train:', X_train.shape)
    print('test:', X_test.shape)

    model_tfidf = make_pipeline(TfidfVectorizer(vocabulary=vocabulary), MultinomialNB())
    model_tfidf.fit(X_train, y_train)
    y_pred_tfidf = model_tfidf.predict(X_test)

    f1 = f1_score(y_test, y_pred_tfidf, average='weighted')
    accuracy = accuracy_score(y_test, y_pred_tfidf)

    print('Multinomial Naive Bayes with TF-IDF:')
    print('-' * 40)
    print(f'f1: {f1:.4f}')
    print(f'accuracy: {accuracy:.4f}')

    report = classification_report(y_test, y_pred_tfidf)

    print('\n Accuracy: ', accuracy_score(y_test, y_pred_tfidf))
    print('\nClassification Report')
    print('======================================================')
    print('\n', report)




llm_folder = "LLM_extractions/results/llm_results"
filenames = [f for f in os.listdir(llm_folder) if f.endswith(".json")]
filenames.sort(key=lambda x : int(x.split('_')[0]))
# filenames = ["0_excerpts.json", "1_excerpts.json", "2_excerpts.json", "3_excerpts.json", "4_excerpts.json"]

nlp = spacy.load("ro_core_news_sm")
data, labels = prepare_data(filenames)
print(data.shape)
train_tf_idf_vectorizer(nlp.Defaults.stop_words, data, labels)