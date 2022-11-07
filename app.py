from flask import Flask,jsonify,request
from flask_cors import CORS
import numpy as np
import pickle


app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST', 'GET'])
def fun():

    product = request.json
    text = product['description']
    loaded_vec = pickle.load(open('model.sav', 'rb'))
    dt = np.array([text])
    p=loaded_vec.transform(dt).toarray()
    loaded_model = pickle.load(open('vec.sav', 'rb'))
    return str(loaded_model.predict(p)[0])  

    

if __name__=='__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)