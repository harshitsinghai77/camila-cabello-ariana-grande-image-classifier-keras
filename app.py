from flask import Flask, redirect, url_for, request, render_template, jsonify
from flask_cors import CORS
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename
import numpy as np
import os
from ariana_camila_classification import *

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return 'Hello World'

@app.route('/predict', methods=[    'POST'])
def upload():
    if request.method == 'POST':
        
        if 'image' not in request.files:
            return jsonify('File not found')
    
        file = request.files['image']
            
        if file and allowed_file(file.filename):
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                            basepath, 'uploads', 'image.jpg')
            file.save(file_path)
            preds = model_predict(file_path)
            return jsonify(str(preds))

        else:
            return jsonify('Bad extension')
        
    return None

if __name__ == '__main__':
    app.run(debug=True)





