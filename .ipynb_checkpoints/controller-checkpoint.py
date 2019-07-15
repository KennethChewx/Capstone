import flask
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

#----- CONFIG -----#
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DEBUG'] = True


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

#-------- MODEL -----------#

df = pd.read_csv('titanic.csv')
include = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Survived']

# Create dummies and drop NaNs
df['Sex'] = df['Sex'].apply(lambda x: 0 if x == 'male' else 1)
df = df[include].dropna()

X = df[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp']]
y = df['Survived']

PREDICTOR = RandomForestClassifier(n_estimators=100).fit(X, y)

















#----- ROUTES -----#
@app.route("/")
def hello():
    return '''
    <body>
    <h2> Hello World! </h2>
    </body>
    '''

@app.route("/pony")
def pony():
    return "My name is pony!"

@app.route('/greet/<name>')
def greet(name):
    '''Say hello to your first parameter'''
    return "Hello, %s!" %name

@app.route('/input_page')
def send_form():
    return flask.render_template('input_page.html')

@app.route('/make_it_happen', methods=['POST'])
def say_hi():
    name = flask.request.form['myname']
    excitement_level = flask.request.form['mylevel']
    
    return flask.render_template('stuff_you_know.html', name=name, lvl=excitement_level)

# This method takes input via an HTML page
@app.route('/titanic_input')
def page():
    return flask.render_template("titanic_input.html")

@app.route('/result', methods=['POST', 'GET'])
def result():
    '''Gets prediction using the HTML form'''
    if flask.request.method == 'POST':
        inputs = flask.request.form
        pclass = inputs['pclass'][0]
        sex = inputs['sex'][0]
        age = inputs['age'][0]
        fare = inputs['fare'][0]
        sibsp = inputs['sibsp'][0]
        
        item = np.array([pclass, sex, age, fare, sibsp]).reshape(1, -1)
        score = PREDICTOR.predict_proba(item)
        results = {'survival chances': score[0,1], 'death chances': score[0,0]}
        return flask.jsonify(results)


@app.route('/predict', methods=["GET"])
def predict():
    pclass = flask.request.args['pclass']
    sex = flask.request.args['sex']
    age = flask.request.args['age']
    fare = flask.request.args['fare']
    sibsp = flask.request.args['sibsp']

    item = np.array([pclass, sex, age, fare, sibsp]).reshape(1,-1)
    score = PREDICTOR.predict_proba(item)
    results = {'survival chances': score[0,1], 'death chances': score[0,0]}
    return flask.jsonify(results)


#----- MAIN SENTINEL -----#
if __name__ == '__main__':
    app.run()