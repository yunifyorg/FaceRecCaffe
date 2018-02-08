from face_features import FaceFeatureExtractor

import numpy as np
import pickle
import requests

from flask import Flask, request, jsonify


app = Flask(__name__)


feature_extractor = FaceFeatureExtractor()


def get_embeddings(face_patches):
    face_patches = np.stack(face_patches)
    feed_dict = {images_placeholder: face_patches, phase_train_placeholder: False}
    return sess.run(embeddings, feed_dict=feed_dict).astype(np.float32)



@app.route('/identify', methods=['POST'])
def identify():
    data = np.fromstring(request.data, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    embs = feature_extractor.getFeatures(img)
    names = []
    distances = []
    if len(emb) > 0:
        for emb in embs:
            data = {'vector': pickle.dumps(emb)}
            r = requests.post('http://localhost:5000/find', json=data)
            r_json = r.json()
            names.append(r_json.get('name', 'Unknown'))
            distances.append(r_json.get('distance', -1))
        return jsonify({'names': names, 'distances': distances})
    else:
        return 'No faces found'
    return jsonify({'vector': pickle.dumps(embs[0])})



@app.route('/add/<name>', methods=['POST'])
def add(name):
    data = np.fromstring(request.data, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    embs = feature_extractor.getFeatures(img)
    
    if len(embs) == 0:
        return 'No face was detected'
    elif len(embs) > 1:
        return 'For registering a name, please use pictures with only one face'

    data = {'name': name,
            'vector': pickle.dumps(embs[0]),
            'model': 'caffe'
            }
    r = requests.post('http://localhost:5000/add', json=data)
    return r.text




if __name__ == "__main__":
    app.run(debug=True, port=8080, host='0.0.0.0')

