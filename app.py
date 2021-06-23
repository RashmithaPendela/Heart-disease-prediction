from flask import Flask, render_template, request

import numpy as np
import pickle
 
app = Flask(__name__) 
 
#load ml model
model=pickle.load(open("log_model.pkl","rb"))

 

# index for the html page
@app.route('/')
def index():
    return render_template("index.html")
 
@app.route('/predict', methods=['POST'])
def predict():
	features=[float(i) for i in request.form.values()]
	array_features=[np.array(features)]
	prediction=model.predict(array_features)
	output=prediction
	if output==1:
		return render_template("index.html",result="patient having heart disease")
	else:
		return render_template("index.html",result="patient don't have heart disease")

# This is where I run the app 
if __name__ == '__main__':
    app.run(debug=True)

