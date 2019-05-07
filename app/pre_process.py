from string import punctuation
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# Pre-processing functions

# If working with TextBlob or VADER it is better not to apply any
# of those pre-processing functions as they might interfere with the
# accuracy of the final result


def remove_sensitivity(text):
    return text.lower()


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', punctuation))


def remove_stopwords(text):
    sw = stopwords.words("english")
    return ' '.join(word for word in tokenize.word_tokenize(text) if word not in sw)


def forced_lem(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    words = tokenize.word_tokenize(text)
    return ' '.join(wordnet_lemmatizer.lemmatize(word) for word in words)


def forced_stem(text):
    porter_stemmer = PorterStemmer()
    words = tokenize.word_tokenize(text)
    return ' '.join(porter_stemmer.stem(word) for word in words)
