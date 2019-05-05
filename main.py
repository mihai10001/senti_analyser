from flask import Flask, render_template, g, jsonify, request

# FLASK SETUP
app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def home():

    if request.method == 'POST':
        if request.form['submit_button'] == 'submit_everything':
            text = request.form.get('text')
            method = request.form.get('onemethod')
            preprocess = request.form.get('onepreprocess')

            if preprocess:
                # import pre_process
                # use pre_process on text
                print('iei')

            if method == 'textblob':
                # from analyser import text_blob
                # dont leave imports here
                # from analyser apply this int(round(2.51*100)) 
                # or better truncate to .3f f and *100 at progressbars :D
                mean_polarity = int(-0.210 * 100)
                polarity = int(-0.234 * 100)
                subjectivity = int(0.656 * 100)
                from analyser import score_degree
                degree = score_degree(polarity)
                return render_template('home.html', method=method, degree=degree, mean_pol=mean_polarity, pol=polarity, subj=subjectivity)
            if method == 'vader':
                mean_polarity = int(0.210 * 100)
                polarity = int(0.234 * 100)
                pos = int(0.656 * 100)
                neg = int(0.455 * 100)
                neu = int(0.123 * 100)
                from analyser import score_degree
                degree = score_degree(polarity)
                return render_template('home.html', method=method, degree=degree, mean_pol=mean_polarity, pol=polarity, pos=pos, neg=neg, neu=neu)
            if method == 'nltk':
                return render_template('home.html', method=method)

    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)