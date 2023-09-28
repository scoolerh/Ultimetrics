import flask
import json

app = flask.Flask(__name__)

@app.route('/')
def welcome():
    return flask.render_template('index.html')

@app.route('/import/')
def fileImport(): 
    return flask.render_template('import.html')

@app.route('/analysis/')
def present():
    return flask.render_template('present.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)