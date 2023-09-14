from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import matplotlib.image as img
import matplotlib.pyplot as pp
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as pp
import os
import base64
from flask import Flask, render_template, jsonify
from flask import request
from flask_cors import CORS 
from mtcnn import MTCNN
import binascii
