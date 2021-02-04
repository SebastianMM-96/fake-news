# Import the neccessary libraries
from flask import Flask, render_template, url_for, request
import pickle

# Set the flask app
app = Flask(
    __name__,
    static_url_path='',
    static_folder='static',
    template_folder='templates'
)

# Load the model
model = pickle.load(open('passiveAgressiveModel.pkl', 'rb'))
# Load the Tf-idf vectorizer
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# # Home page
@app.route('/')
def index():
    return render_template('index.html')

# App page
@app.route('/tantei')
def tantei():
    return render_template('tantei.html')

@app.route('/result')
def result():
    return render_template('result.html')

# Predicted result
@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    # Fix this part
    predict = model.predict(tfidf.transform[message])

    return render_template('result.html', prediction = predict)

# Running in port 3200
if __name__ == '__main__':
    # Run the app
    app.run(port = 3200, debug = True)