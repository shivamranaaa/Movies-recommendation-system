from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import joblib

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process

model_knn=pickle.load(open('model_knn.pkl','rb'))
final=pickle.load(open('final.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		movie_name = request.form['movie_name']

	return render_template('index.html',prediction = final)

if __name__ == '__main__':
	app.run()