import os
from flask import Flask, render_template, request

# Preprocess
from preprocess import preprocess

# Analyzers
from analyser import score_degree
from analyser import textblob_method
from analyser import vader_method
from analyser import nltk_ml_method

# Nltk data and corpus
import nltk
nltk.download('punkt')

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

            if text:
                if method == 'textblob':

                    if preprocess == method:
                        preprocess_list = request.form.getlist('preprocess-list')
                        text = preprocess(text, preprocess_list)
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

                    if preprocess == method:
                        preprocess_list = request.form.getlist('preprocess-list')
                        text = preprocess(text, preprocess_list)
                    overall_sentiment, mean_polarity, raw_data = vader_method(text)
                    mean_polarity = round(mean_polarity, 3)
                    polarity = round(overall_sentiment['compound'], 3)
                    pos = round(overall_sentiment['pos'], 3)
                    neg = round(overall_sentiment['neg'], 3)
                    neu = round(overall_sentiment['neu'], 3)
                    degree = score_degree(polarity)
                    if polarity >= 0.05:
                        color = 'success'
                    elif polarity <= -0.05:
                        color = 'danger'
                    else:
                        color = 'dark'

                    return render_template('home.html', method=method, color=color, degree=degree, mean_pol=mean_polarity, pol=polarity, pos=pos, neg=neg, neu=neu, adv=advanced, raw=raw_data)

                elif method == 'nltk':

                    if preprocess == method:
                        preprocess_list = request.form.getlist('preprocess-list')
                        preprocess_list.extend(['Uncapitalize', 'Remove stop words', 'Lemmatization', 'Stemming'])
                        text = preprocess(text, preprocess_list)
                    overall_sentiment, mean_polarity, raw_data = nltk_ml_method(text)
                    degree = score_degree(mean_polarity)
                    if mean_polarity >= 0.05:
                        color = 'success'
                    elif mean_polarity <= -0.05:
                        color = 'danger'
                    else:
                        color = 'dark'

                    return render_template('home.html', method=method, color=color, degree=degree, mean_pol=mean_polarity, pol=overall_sentiment, adv=advanced, raw=raw_data)

                else:
                    return render_template('home.html', error='No analyse method chosen')

            else:
                return render_template('home.html', error='No text input to analyze. Input some text first')


    return render_template('home.html')

if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=False, port=os.environ.get('PORT'))
