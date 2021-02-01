# Import the neccessary libraries
from flask import Flask, render_template, url_for

# Set the flask app
app = Flask(
    __name__,
    static_url_path='',
    static_folder='static',
    template_folder='templates'
)

# Home page
@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')

# App page
@app.route('/tantei', methods = ['GET', 'POST'])
def tantei():
    return render_template('tantei.html')

# Running in port 3200
if __name__ == '__main__':
    app.run(port = 3200, debug = True)