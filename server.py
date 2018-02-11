import matplotlib
matplotlib.use("Pdf")

from face_features import FaceFeatureExtractor

import os
import numpy as np
import pickle
import requests
import cv2

from flask import Flask, request, jsonify, render_template


app = Flask(__name__)


feature_extractor = FaceFeatureExtractor()
DB_ADDRESS = os.environ.get('DB_ADDRESS', 'http://localhost:5000')


def get_embeddings(face_patches):
    face_patches = np.stack(face_patches)
    feed_dict = {images_placeholder: face_patches, phase_train_placeholder: False}
    return sess.run(embeddings, feed_dict=feed_dict).astype(np.float32)



@app.route('/identify', methods=['POST'])
def identify():
    json = request.get_json()
    data = np.fromstring(json['img'].decode('base64'), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    print(img.shape)
    embs, times = feature_extractor.getFeatures(img)
    names = []
    distances = []
    filenames = []
    if len(embs) > 0:
        for emb in embs:
            data = {'vector': pickle.dumps(emb), 'model': 'caffe'}
            r = requests.post(DB_ADDRESS + '/find', json=data)
            r_json = r.json()
            names.append(r_json.get('name', 'Unknown'))
            distances.append(r_json.get('distance', -1))
            filenames.append(r_json.get('filename'))
        return jsonify({'names': names, 'distances': distances, 'times': times, 'filenames': filenames})
    else:
        return 'No faces found'



@app.route('/add', methods=['POST'])
def add():
    json = request.get_json()
    print('Should print now')
    data = np.fromstring(json['img'].decode('base64'), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    print(img.shape)
    embs, times = feature_extractor.getFeatures(img)
    
    if len(embs) == 0:
        return 'No face was detected'
    elif len(embs) > 1:
        return 'For registering a name, please use pictures with only one face'

    data = {'name': json['name'],
            'vector': pickle.dumps(embs[0]),
            'model': 'caffe',
            'filename': json['filename']
            }
    r = requests.post(DB_ADDRESS + '/add', json=data)
    return r.text


@app.route('/app', methods=['GET'])
def webapp():
    return render_template('app.html') 




if __name__ == "__main__":
    app.run(debug=True, port=8080, host='0.0.0.0', ssl_context=('cert.pem', 'key.pem'))

