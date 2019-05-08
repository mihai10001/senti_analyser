from statistics import mean
from nltk import tokenize

# TextBlob
from textblob import TextBlob
# VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# NLTK


# Calculate the polarity of the paragraph usign the TextBlob library
# TextBlob's trained classifier uses sentences as input
# Tokenization is done automatically when creating a TextBlob object
# Each sentence is assigned a polarity and subjectivity value
# Also return the raw data
# My approach: calculate the mean of all sentence polarities

# Features:
# Tokenizer: "If None, defaults to WordTokenizer()"
# Analyzer:  "If None, defaults to PatternAnalyzer"
# TextBlob has different ratings depending on the form of the word and therefore the input should not be stemmed or lemmatized
# TextBlob uses sentences as tokens, so it's prefered not to remove punctuation as that affects TextBlob's polarity calculations

def textblob_method(paragraph):
    blob = TextBlob(paragraph)

    overall_sentiment = [blob.sentiment.polarity, blob.sentiment.subjectivity]
    my_mean_sentiment = mean(sentence.sentiment.polarity for sentence in blob.sentences)
    raw_data = [[sentence.raw.strip()[:30] + ' ...', sentence.polarity, sentence.subjectivity] for sentence in blob.sentences]

    return overall_sentiment, my_mean_sentiment, raw_data


# Calculate the polarity of the paragraph usign the VADER library
# VADER's trained classifier also uses sentences as input
# Tokenization can be done manually by using ntlk tokenize module
# Each sentence is assigned a compound(overall),a positive,a negative and a neutral value
# Also return the raw data
# My approach: calculate the mean of all sentence polarities

# Features:
# Tokenizer: manually selected
# Analyzer:  SentimentIntensityAnalyzer
# VADER already removes stopwords
# VADER has different ratings depending on the form of the word and therefore the input should not be stemmed or lemmatized
# VADER uses sentences as tokens, so it's prefered not to remove punctuation as that affects VADER's polarity calculations

def vader_method(paragraph):
    sid = SentimentIntensityAnalyzer()
    sentences = tokenize.sent_tokenize(paragraph)

    order_of_keys = ['compound', 'pos', 'neg', 'neu']
    overall_sentiment = sid.polarity_scores(paragraph)
    my_mean_sentiment = mean(sid.polarity_scores(sentence)['compound'] for sentence in sentences)
    raw_data = [[sentence.strip()[:30] + '...', *[sid.polarity_scores(sentence)[key] for key in order_of_keys]] for sentence in sentences]

    return overall_sentiment, my_mean_sentiment, raw_data


# Calculate the polarity of the paragraph usign an NLTK trained classifier
# Classifier trained manually based on movie reviews data downloaded from NLTK
# Classifier trained on words as tokens, so tokenization will be kept at word level
# Classifier trained in order to give a pos/neg status about the word, based on the likelyhood of
# that word being positive or negative in the given data set
# My approach: calculate the mean of all words polarities, taking into account the ones that do not
# appear in the data set

# Features:
# A whole lot of features
def nltk_ml_method(paragraph):
    pass



def score_degree(score):
    if 0.75 <= score < 1:
        return 'Very positive'
    elif 0.25 <= score < 0.75:
        return 'Positive'
    elif 0.05 <= score < 0.25:
        return 'Pretty positive'
    elif -0.05 <= score < 0.05:
        return 'Neutral'
    elif -0.25 <= score < -0.05:
        return 'Pretty negative'
    elif -0.75 <= score < -0.25:
        return 'Negative'
    elif -1 <= score < -0.75:
        return 'Very negative'


test_text = '''
The titular threat of The Blob has always struck me as the ultimate movie
monster: an insatiably hungry, amoeba-like mass able to penetrate
virtually any safeguard, capable of--as a doomed doctor chillingly
describes it--"assimilating flesh on contact.
Snide comparisons to gelatin be damned, it's a concept with the most
devastating of potential consequences, not unlike the grey goo scenario
proposed by technological theorists fearful of
artificial intelligence run rampant.
'''
