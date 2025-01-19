from sklearn.utils import shuffle
import os, json
import spacy.lang
from util_functions import clean_leaflet
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
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

    # take only some of the sentences labeled 0, so as to be proportional to the sentences labeled 1
    sentences = np.random.choice(sentences, sentences_1.shape[0])
    labels = [0 for _ in range(sentences.shape[0])]

    return sentences, labels

def prepare_data(filenames):
    sentences_1, labels_1, sentences_1_mean_len, sentences_1_std_len = prepare_data_with_label_1(filenames)
    sentences_0, labels_0 = prepare_data_with_label_0(filenames, sentences_1, sentences_1_mean_len, sentences_1_std_len)
    
    with open('LLM_extractions/results/sentence_complexity_classification_results_stop_words.txt', 'w') as file:
        file.write('Sentences labeled 0: ' + str(sentences_0.shape[0]) + '\n')
        file.write('Sentences labeled 1: ' + str(sentences_1.shape[0]) + '\n')
    with open('LLM_extractions/results/sentence_complexity_classification_results_pos.txt', 'w') as file:
        file.write('Sentences labeled 0: ' + str(sentences_0.shape[0]) + '\n')
        file.write('Sentences labeled 1: ' + str(sentences_1.shape[0]) + '\n')
    print('Sentences labeled 0: ', sentences_0.shape[0])
    print('Sentences labeled 1: ', sentences_1.shape[0])
    
    data = np.concatenate((sentences_1, sentences_0), axis=0)
    labels = labels_1 + labels_0

    # shuffle
    data, labels = shuffle(data, labels)

    return data, labels


def translate_to_pos(data):
    pos_data = []
    for sentence in data:
        tokens = nlp(str(sentence))
        translated_sentence = ' '.join([token.pos_ for token in tokens])
        pos_data.append(translated_sentence)

    return np.array(pos_data)


def train_tf_idf_vectorizer(data, labels, results_filename, vocabulary=None):
    with open(results_filename, 'a') as file:
        
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 0)

        file.write('\nTrain data: ' + str(X_train.shape[0]))
        file.write('\nTest data: ' + str(X_test.shape[0]) + '\n')
        print('train:', X_train.shape)
        print('test:', X_test.shape)

        if vocabulary is None:
            model_tfidf = make_pipeline(TfidfVectorizer(), MultinomialNB())
        else:
            model_tfidf = make_pipeline(TfidfVectorizer(vocabulary=vocabulary), MultinomialNB())

        model_tfidf.fit(X_train, y_train)
        y_pred_tfidf = model_tfidf.predict(X_test)

        report = classification_report(y_test, y_pred_tfidf)

        file.writelines(['\nAccuracy: ', str(accuracy_score(y_test, y_pred_tfidf))])
        file.write('\n\nClassification Report\n')
        file.write('======================================================')
        file.writelines(['\n', report])
        print('\n Accuracy: ', accuracy_score(y_test, y_pred_tfidf))
        print('\nClassification Report')
        print('======================================================')
        print('\n', report)




llm_folder = "LLM_extractions/results/llm_results"
filenames = [f for f in os.listdir(llm_folder) if f.endswith(".json")]
filenames.sort(key=lambda x : int(x.split('_')[0]))

nlp = spacy.load("ro_core_news_sm")
data, labels = prepare_data(filenames)
print('Data shape', data.shape)

# classify using stop words
train_tf_idf_vectorizer(data, labels, 'LLM_extractions/results/sentence_complexity_classification_results_stop_words.txt', nlp.Defaults.stop_words)

# classify using pos
translated_data = translate_to_pos(data)
train_tf_idf_vectorizer(translated_data, labels, 'LLM_extractions/results/sentence_complexity_classification_results_pos.txt')