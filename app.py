import json
from flask import Flask 
from flask import request, Response, jsonify

from sentiment_analyzer import analyze_sentiment

app = Flask(__name__)

# base URL
@app.route('/')
def index():
    return "This is an API for sentiment analysis. Please use POST method on /analyze"

# API endpoint
@app.route('/analyze', methods = ['POST'])
def predict_sentiment():
    try:
        # get promt text
        promt_text = request.json['text']
        # get analysis result from the sentiment analysis model
        result = analyze_sentiment(promt_text)
        # if the result is an exception then return correct response
        if isinstance(result, Exception):
            res = {
                'error': str(result)
            }
            res = Response(json.dumps(res), status=513, mimetype='application/json')
            return res
        # return the result as response
        return jsonify({
            "sentiment" : result
            })
    except Exception as e:
        res = {
            'error': repr(e)
        }
        res = Response(json.dumps(res), status=500, mimetype='application/json')
        return res

if __name__ == '__main__':
    app.run(debug=True)