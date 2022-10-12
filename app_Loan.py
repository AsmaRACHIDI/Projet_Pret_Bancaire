from flask import Flask,render_template,request,url_for
import pickle
import numpy as np
from regex import E
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
import sklearn


#instantiate flask

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def hello():
    return render_template('home.html')

@app.route('/predict',methods=["POST", "GET"])
def predict():
    feature=[float(x) for x in request.form.values()]
    ### Scaler mes valeurs
    e=[[0, 1, 2, 1, 0, 6344443, 4754440, 130, 360, 1, 1]]
    #print('The scikit-learn version is {}.'.format(sklearn.__version__))
    print(e)
    feature_final=np.array(e).reshape(1,-1)
    print(feature_final)
    
    prediction=model.predict(feature_final)
    #prediction=model.predict(e)
    print(prediction)
    return render_template('home.html',prediction_text="Réponse = {}".format(str(prediction)))

if __name__ == '__main__':
    app.run()