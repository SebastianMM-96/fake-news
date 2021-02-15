from flask import Flask, render_template, url_for, request
import pickle

# Set the flask app
app = Flask(
    __name__,
    static_url_path='',
    static_folder='static',
    template_folder='templates'
)

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# App page
@app.route('/tantei')
def tantei():
    return render_template('tantei.html')

# Prediction
@app.route('/predict')
def predict():
    message = request.form['message']
    myPrediction = model.predict(tv.transform([message]))
    return render_template('tantei.html', prediction = myPrediction)

# Run the app
if __name__== '__main__':
    model = pickle.load(open('model.pkl', 'rb'))
    tv = pickle.load(open('tv.pkl', 'rb'))
    # Run the app
    app.run(debug = True)