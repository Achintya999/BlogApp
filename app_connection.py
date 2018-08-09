from flask import Flask
from flask import jsonify, request

from sklearn.externals import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

import string

app=Flask(__name__)


@app.route('/',methods=['POST'])
def pred():
	classifier=joblib.load('XGBOOST.pkl')
	msg1 = request.json['msg']
	String1 = classifier.predict([msg1])
	
	return jsonify({"result":String1[0]}),220
	
if __name__ == "__main__":
		app.run(port=8000,debug=True)