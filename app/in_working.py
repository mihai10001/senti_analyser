# INSPIRATION
# https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ
# https://towardsdatascience.com/basic-binary-sentiment-analysis-using-nltk-c94ba17ae386
# https://www.nltk.org/book/ch06.html
import nltk.classify.util
from nltk.probability import FreqDist
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk import tokenize


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

    # Create and train a classifier. Return it and the accuracy
    classifier = NaiveBayesClassifier.train(training_set)
    accuracy = nltk.classify.accuracy(classifier, testing_set) * 100
    return classifier, accuracy


def short_reviews_classifier():
    # Short reviews data
    positive_reviews = open('train_data/positive.txt', 'r', encoding='latin2').read()
    negative_reviews = open('train_data/negative.txt', 'r', encoding='latin2').read()

    pos_doc = [(rev, 'pos') for rev in positive_reviews.split('\n')]
    neg_doc = [(rev, 'neg') for rev in negative_reviews.split('\n')]
    documents = [pos_doc, neg_doc]

    all_words = []
    for w in tokenize.word_tokenize(positive_reviews):
        all_words.append(w.lower())
    for w in tokenize.word_tokenize(negative_reviews):
        all_words.append(w.lower())

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

    # Create and train a classifier. Return it and the accuracy
    classifier = NaiveBayesClassifier.train(training_set)
    accuracy = nltk.classify.accuracy(classifier, testing_set) * 100
    return classifier, accuracy
