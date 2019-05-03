from textblob import TextBlob
from statistics import mean

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import tokenize

# Calculate the polarity of the paragraph usign the TextBlob library
# TextBlob's trained classifier uses sentences as input
# Tokenization is done automatically when creating a TextBlob object
# Each sentence is assigned a polarity and subjectivity value
# Also return the raw data
# My approach: calculate the mean of all sentence polarities
def textblob_method(paragraph):
    blob = TextBlob(paragraph)
    
    # Stemminzation, Lemmitization, other processes go here

    my_polarity_mean = mean(sentence.sentiment.polarity for sentence in blob.sentences)
    raw_data = [blob.sentiment]
    raw_data.append([(sentence.strip()[:15] + '...', sentence.sentiment) for sentence in blob.sentences])

    return my_polarity_mean, raw_data


# Calculate the polarity of the paragraph usign the vader library
# vader's trained classifier also uses sentences as input
# Tokenization can be done manually by using ntlk tokenize module
# Each sentence is assigned a compound(overall),a positive,a negative and a neutral value
# Also return the raw data
# My approach: calculate the mean of all sentence polarities
def vader_method(paragraph):
    sid = SentimentIntensityAnalyzer()
    sentences = tokenize.sent_tokenize(paragraph)

    # Stemminzation, Lemmitization, other processes go here

    my_polarity_mean = mean(sid.polarity_scores(sentence)['compound'] for sentence in sentences)
    raw_data = [sid.polarity_scores(paragraph)]
    raw_data.append([[sentence.strip()[:15] + '...', sid.polarity_scores(sentence)] for sentence in sentences])

    return my_polarity_mean, raw_data


# A score between -0.05 and 0.05 is classified as neutral
def score_str(score):
    if score >= 0.05: 
        return "Positive"
    elif score <= - 0.05: 
        return "Negative"
    else: 
        return "Neutral"


text = '''
The titular threat of The Blob has always struck me as the ultimate movie
monster: an insatiably hungry, amoeba-like mass able to penetrate
virtually any safeguard, capable of--as a doomed doctor chillingly
describes it--"assimilating flesh on contact.
Snide comparisons to gelatin be damned, it's a concept with the most
devastating of potential consequences, not unlike the grey goo scenario
proposed by technological theorists fearful of
artificial intelligence run rampant.
''' 

score, raw_data = textblob_method(text)
print("The overall vader score is: {:.5f} points, assesed {}".format(score, score_str(score)))
for x in raw_data:
    print(x)

print('\n')

score, raw_data = vader_method(text)
print("The overall vader score is: {:.5f} points, assesed {}".format(score, score_str(score)))
for x in raw_data:
    print(x)