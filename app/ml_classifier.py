# INSPIRATION
# https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ
# https://towardsdatascience.com/basic-binary-sentiment-analysis-using-nltk-c94ba17ae386
# https://www.nltk.org/book/ch06.html
import random
import pickle
import nltk.classify.util
from nltk.probability import FreqDist
from nltk.classify import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import movie_reviews
from nltk import tokenize
from sklearn.linear_model import SGDClassifier

def movie_reviews_classifier():
    # Movie reviews data
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]

    # Get all words lowered
    all_words = [w.lower() for w in movie_reviews.words()]

    # Frequently distributed dict ensures unicity across found words
    dist = FreqDist(all_words)

    # Pick the 3000 most frequent words
    word_features = list(dist.keys())[:3000]

    # Function to create a dictionary of features for each review in the list document.
    # The keys are the words in word_features
    # The values of each key are either true or false whether that feature appears in the review or not
    def find_features(document):
        words = set(document)
        features = {}
        for w in word_features:
            features[w] = (w in words)
        return features

    # Find features of all the reviews
    featuresets = [(find_features(rev), category) for (rev, category) in documents]

    # Select the training and testing sets
    training_set = featuresets[:1500]
    testing_set = featuresets[1500:]

    # Train a prefered classifier. Return the accuracy
    classifier = NaiveBayesClassifier.train(training_set)
    accuracy = nltk.classify.accuracy(classifier, testing_set) * 100
    return accuracy


def short_movie_reviews_classifier():
    # Short reviews datas et
    positive_reviews = open('train_data/positive.txt', 'r', encoding='latin2').read()
    negative_reviews = open('train_data/negative.txt', 'r', encoding='latin2').read()

    pos_doc = [(rev, 'pos') for rev in positive_reviews.split('\n')]
    neg_doc = [(rev, 'neg') for rev in negative_reviews.split('\n')]
    documents = [*pos_doc, *neg_doc]
    
    # Create a list of words found in the data set, and ensure they are allowed
    # J is adjective, R is adverb, and V is verb
    all_words = []
    allowed_word_types = ['J', 'R', 'V']
   
    for (rev, status) in pos_doc:
        words = tokenize.word_tokenize(rev)
        pos = nltk.pos_tag(words)
        for w in pos:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())

    for (rev, status) in neg_doc:
        words = tokenize.word_tokenize(rev)
        pos = nltk.pos_tag(words)
        for w in pos:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())

    # Frequently distributed dict ensures unicity across found words
    dist = FreqDist(all_words)

    # Pick the 5000 most frequent words
    # save_pickle(word_features, path='train_data/word_features')
    word_features = list(dist.keys())[:5000]

    # Function to create a dictionary of features for each review in the list document.
    # The keys are the words in word_features
    # The values of each key are either true or false whether that feature appears in the review or not
    def find_features(document):
        words = tokenize.word_tokenize(document)
        features = {}
        for w in word_features:
            features[w] = (w in words)
        return features

    # Find features of all the reviews
    feature_sets = [(find_features(rev), category) for (rev, category) in documents]
    # Randomize featuresets to distribute positive and negative data evenly
    random.shuffle(feature_sets)

    # Select the training and testing sets
    training_set = feature_sets[:10000]
    testing_set = feature_sets[10000:]

    # Train a prefered classifier. Return the accuracy
    # SGDclassifier = SklearnClassifier(SGDClassifier())
    # SGDclassifier.train(training_set)
    # save_pickle(classifier, path='train_data/classifier')
    classifier = NaiveBayesClassifier.train(training_set)
    accuracy = nltk.classify.accuracy(SGDclassifier, testing_set) * 100
    return accuracy


# Just a simple find_features function wrapper so I can classify without
# the need of computing any data other than just the result of the classification
def features_wrapper(document):
    words = tokenize.word_tokenize(document)
    word_features = get_pickle(path='train_data/word_features')
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


def save_pickle(data, path):
    with open(path, 'wb') as objbytes:
        pickle.dump(data, objbytes)


def get_pickle(path):
    with open(path, 'rb') as objbytes:
        data = pickle.load(objbytes)
    return data


def classify(text, classifier):
    features = features_wrapper(text)
    if True in features.values():
        return 1 if classifier.classify(features) == 'pos' else -1
    else:
        return 0


# 67.46987951807229 SGD
# 68.97590361445783 NB

