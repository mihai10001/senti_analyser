from flask import Flask, render_template, request

#Preprocess
from pre_process import remove_sensitivity, remove_punctuation, remove_stopwords, forced_lem, forced_stem

# Analyzers
from analyser import score_degree
from analyser import textblob_method
from analyser import vader_method

# FLASK SETUP
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():

    if request.method == 'POST':
        if request.form['submit_button'] == 'submit_everything':
            text = request.form.get('text')
            preprocess = request.form.get('preprocess')
            method = request.form.get('onemethod')
            advanced = request.form.get('advanced')

            if preprocess:
                preprocess_list = request.form.getlist('preprocess-list')
                if 'Uncapitalize' in preprocess_list:
                    text = remove_sensitivity(text)
                if 'Remove punctuation' in preprocess_list:
                    text = remove_punctuation(text)
                if 'Remove stop words' in preprocess_list:
                    text = remove_stopwords(text)
                if 'Lemmatization' in preprocess_list:
                    text = forced_lem(text)
                if 'Stemming' in preprocess_list:
                    text = forced_stem(text)

            if method == 'textblob':
                overall_sentiment, mean_polarity, raw_data = textblob_method(text)
                mean_polarity = round(mean_polarity, 3)
                polarity = round(overall_sentiment[0], 3)
                subjectivity = round(overall_sentiment[1], 3)
                degree = score_degree(polarity)
                if polarity >= 0.05:
                    color = 'success'
                elif polarity <= -0.05:
                    color = 'danger'
                else:
                    color = 'dark'

                return render_template('home.html', method=method, color=color, degree=degree, mean_pol=mean_polarity, pol=polarity, subj=subjectivity, adv=advanced, raw=raw_data)

            elif method == 'vader':
                overall_sentiment, mean_polarity, raw_data = vader_method(text)
                mean_polarity = round(mean_polarity, 3)
                polarity = round(overall_sentiment['compound'], 3)
                pos = round(overall_sentiment['pos'], 3)
                neg = round(overall_sentiment['neg'], 3)
                neu = round(overall_sentiment['neu'], 3)
                degree = score_degree(polarity)
                if polarity >= 0:
                    color = 'success'
                else:
                    color = 'danger'

                return render_template('home.html', method=method, color=color, degree=degree, mean_pol=mean_polarity, pol=polarity, pos=pos, neg=neg, neu=neu, adv=advanced, raw=raw_data)

            elif method == 'nltk':
                return render_template('home.html', method=method, color=color, degree=degree, adv=advanced, raw=raw_data)

    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
