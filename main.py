from flask import Flask, render_template, g, jsonify, request

# FLASK SETUP
app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def home():

    if request.method == 'POST':
        if request.form['submit_button'] == 'submit_everything':
            text = request.form.get('text')
            return render_template('home.html', text=text)

    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)