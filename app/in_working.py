# INSPIRATION 
# https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ
# https://towardsdatascience.com/basic-binary-sentiment-analysis-using-nltk-c94ba17ae386

def nltk_method(paragraph):
    # from nltk.classify import NaiveBayesClassifier
    from nltk.corpus import movie_reviews
    # from nltk.sentiment import SentimentAnalyzer
    

    # n_instances = 100
    # subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
    # obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]
    # len(subj_docs), len(obj_docs)

    # # Each document is represented by a tuple (sentence, label). The sentence is tokenized, so it is represented by a list of strings:
    # # We separately split subjective and objective instances to keep a balanced uniform class distribution in both train and test sets.

    # train_subj_docs = subj_docs[:80]
    # test_subj_docs = subj_docs[80:100]
    # train_obj_docs = obj_docs[:80]
    # test_obj_docs = obj_docs[80:100]
    # training_docs = train_subj_docs+train_obj_docs
    # testing_docs = test_subj_docs+test_obj_docs

    # documents = [(list(movie_reviews.words(fileid)), category)
    #             for category in movie_reviews.categories()
    #             for fileid in movie_reviews.fileids(category)]
    all_words = []

    for w in movie_reviews.words():
        all_words.append(w.lower())

    # Frequently distributed dict ensures unicity across found words
    # Most common 1500 words of the language, there is an average of around ~3000 used words in a language at any time
    dist = nltk.FreqDist(all_words)

    # listing the 5000 most frequent words
    word_features = list(dist.keys())[:1500]

    # function to create a dictionary of features for each review in the list document.
    # The keys are the words in word_features 
    # The values of each key are either true or false for wether that feature appears in the review or not

    def find_features(document):
        words = word_tokenize(document)
        features = {}
        for w in word_features:
            features[w] = (w in words)

    # sentim_analyzer = SentimentAnalyzer()
    # all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])
    # # We use simple unigram word features, handling negation:

    # unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
    # len(unigram_feats)

    # sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
    # # We apply features to obtain a feature-value representation of our datasets:

    # training_set = sentim_analyzer.apply_features(training_docs)
    # test_set = sentim_analyzer.apply_features(testing_docs)
    # # We can now train our classifier on the training set, and subsequently output the evaluation results:

    # trainer = NaiveBayesClassifier.train
    # classifier = sentim_analyzer.train(trainer, training_set)
    # # Training classifier

    # for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
    #     print('{0}: {1}'.format(key, value))

    # Evaluating NaiveBayesClassifier results...
    # Accuracy: 0.8
    # F-measure [obj]: 0.8
    # F-measure [subj]: 0.8
    # Precision [obj]: 0.8
    # Precision [subj]: 0.8
    # Recall [obj]: 0.8
    # Recall [subj]: 0.8

