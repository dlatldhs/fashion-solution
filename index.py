from flask import Flask
import pose_media
import matplotlib.image as img
import matplotlib.pyplot as pp

import numpy as np
from google.colab.patches import cv2_imshow
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, My First Flask!'

@app.route('/analysis')
def user_body_analysis():

    # TODO use cam and take a picture


    # TODO throw picture
    img_result, result = pose_media.get_prediction(img)
    pp.imshow(img_result)
    pp.show()

    print(f"result: 'upper':{result[0]},'bottom':{result[1]}")
    # return jsonify({'upper': result[0]},{'bottom':result[1]})