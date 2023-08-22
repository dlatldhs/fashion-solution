import cv2
import tempfile
import os
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import pose
from mediapipe.python.solutions.pose import Pose

import numpy as np
from google.colab.patches import cv2_imshow
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.image as img
import matplotlib.pyplot as pp

def get_prediction(image_bytes):
    pose = Pose()
    mp_drawing = mp.solutions.drawing_utils
    # 이미지 파일로 저장
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(image_bytes)
    temp_file.close()

    # 저장한 이미지 파일 경로
    image_path = temp_file.name

    # OpenCV로 이미지 읽기
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 머리 위치 저장
    is_first = True
    first_center_x,first_center_y , first_radius = None,None,None # 얼굴 x , y 좌표 및 얼굴 반지름

    # 이미지 세부 값 저장
    # image value save
    img_h, img_w, _ = img.shape
    img_result = img.copy()

    # image bgr -> image rgb transform
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # put image into the pose model
    results = pose.process(img)

    # 관절 위치 drawing
    mp_drawing.draw_landmarks(
        img_result,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )

    # 랜드마크 그리기
    # landmarks is avaliable
    if results.pose_landmarks:

      # recevice landmark
      landmark = results.pose_landmarks.landmark
      LEFT_SHOULDER_x = landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * img_w
      LEFT_SHOULDER_y = landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * img_h

      RIGHT_SHOULDER_x = landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * img_w
      RIGHT_SHOULDER_y = landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * img_h

      SHOULDER_LENGTH = LEFT_SHOULDER_x - RIGHT_SHOULDER_x
      # 어깨 길이 측정
      # 예외처리
      SHOULDER = False
      if SHOULDER_LENGTH > 0 and SHOULDER_LENGTH <= 100:
        print("제대로 서서 전신이 사진에 나오게 다시 찍어주세요!")
        SHOULDER = 1
      elif SHOULDER_LENGTH < 140:
        SHOULDER = 2 # 어좁이
      elif SHOULDER_LENGTH >= 140 and SHOULDER_LENGTH <= 150:
        SHOULDER = 3 # 보통 어깨
      elif SHOULDER_LENGTH >= 155 and SHOULDER_LENGTH <= 160:
        SHOULDER = 4 # 조금 넓음
      elif SHOULDER_LENGTH >= 161 and SHOULDER_LENGTH <= 180:
        SHOULDER = 5 # 어깡
      elif SHOULDER_LENGTH >= 181 and SHOULDER_LENGTH <= 200:
        SHOULDER = 6 # 신
      else:
        SHOULDER = 7 # 측정불가 or 예외값
        print(SHOULDER_LENGTH)

      # 하체 길이 측정
      LEFT_HIP_y = landmark[mp_pose.PoseLandmark.LEFT_HIP].y * img_h
      RIGHT_HIP_y = landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * img_h
      # 발 y 좌표
      RIGHT_FOOT_INDEX_y = landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * img_h
      LEFT_FOOT_INDEX_y = landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * img_h

      SUM_HIP_LOCATE = (LEFT_HIP_y + RIGHT_HIP_y)/2

      # 어깨 y
      SUM_SHOULDER_LOCATE = (LEFT_SHOULDER_y+RIGHT_SHOULDER_y)/2

      # 발 y 좌표
      SUM_FOOT_LOCATE = (LEFT_FOOT_INDEX_y+RIGHT_FOOT_INDEX_y)/2 # 1018

      # 상체 + 하체 길이
      ALL_LENGTH = SUM_FOOT_LOCATE - SUM_SHOULDER_LOCATE

      # 상체
      UPPER_BODY = SUM_HIP_LOCATE - SUM_SHOULDER_LOCATE

      # 하체
      BOTTOM_BODY = SUM_FOOT_LOCATE - SUM_HIP_LOCATE

      # 비율 계산법  => 특정값 / 전체값 * 100 = n%

      upper_criteria = 54.3215
      bottom_criteria = 45.6875
      plus_value = 5

      upper_long = False
      upper_short= False
      upper_very_long = False
      upper_very_short= False

      bottom_long = False
      bottom_short= False
      bottom_very_long = False
      bottom_very_short= False

      # 54.3125
      upper_body_ratio = UPPER_BODY/ALL_LENGTH*100

      result = []

      # upper_criteria < user_upper_body < upper_criteria+5
      if upper_body_ratio > upper_criteria and upper_body_ratio < upper_criteria+plus_value:
        print("상체가 평균보다 긴 편")
        upper_long = True
        result.append("upper_long")
      elif upper_body_ratio < upper_criteria and upper_body_ratio > upper_criteria-plus_value:
        print("상체가 평균보다 짧은 편")
        upper_short = True
        result.append("upper_short")
      elif upper_body_ratio < upper_criteria:
        print("상체가 평균보다 많이 짧은 편")
        upper_very_short = True
        result.append("upper_very_short")
      elif upper_body_ratio > upper_criteria:
        upper_very_long = True
        result.append("upper_very_long")
        print("상체가 평균보다 많이 긴 편")

      # 45.6875
      bottom_body_ratio = BOTTOM_BODY/ALL_LENGTH*100

      # bottom_criteria < user_bottom_body < bottom_criteria+5
      if bottom_body_ratio > bottom_criteria and bottom_body_ratio < bottom_criteria+plus_value:
        print("하체가 평균보다 긴 편")
        bottom_long = True
        result.append("bottom_long")
      elif bottom_body_ratio < bottom_criteria and bottom_body_ratio > bottom_criteria-plus_value:
        print("하체가 평균보다 짧은 편")
        bottom_short = True
        result.append("bottom_short")
      elif bottom_body_ratio < bottom_criteria:
        print("하체가 평균보다 많이 짧은 편")
        bottom_very_short = True
        result.append("bottom_very_short")
      elif bottom_body_ratio > bottom_criteria:
        print("하체가 평균보다 많이 긴 편")
        bottom_very_long = True
        result.append("bottom_very_long")

      print(f"몸 길이 : {ALL_LENGTH}, 상체 : {UPPER_BODY}, 하체 : {BOTTOM_BODY}")
    # 임시 파일 삭제
    os.unlink(image_path)
    # 결과 반환
    return img_result,result