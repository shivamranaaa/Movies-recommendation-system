from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import joblib

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process

mat_movies_users = pickle.load(open("mat_movies_users.pkl", 'rb'))
model_knn=pickle.load(open('model_knn.pkl','rb'))
df_movies=pickle.load(open('df_movies.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		movie_name = request.form['movie_name']

		model_knn.fit(mat_movies_users)

		idx=process.extractOne(movie_name, df_movies['title'])[2]
		distances, indices=model_knn.kneighbors(mat_movies_users[idx], n_neighbors=6)
		
		x=[]
		for i in indices:
			x.append(df_movies['title'][i].where(i!=idx))
		df=pd.DataFrame(x)
		
		y=[]
		for i,j in df.iloc[:,:].items():
			y.append(j)
		final=[i[0] for i in y]
		final=final[1:]
		
	return render_template('index.html',prediction = final)

if __name__ == '__main__':
	app.run()