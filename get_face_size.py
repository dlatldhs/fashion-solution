'''
import cv2
import mediapipe as mp

# MediaPipe 얼굴 감지 및 부속 요소 초기화
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection()

# 이미지 읽기
img = cv2.imread("images.jpg")

# RGB로 변환
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 이미지 처리 및 얼굴 감지
results = face_detection.process(rgb_img)

# 검출된 얼굴에 대해 박스 그리기 및 크기 출력
if results.detections:
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = img.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        print(f"얼굴 크기: {w}x{h}")

# 이미지 출력
cv2.imshow("Faces Detected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

import cv2

# Haar Cascade 분류기 불러오기
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# 이미지 읽기
img = cv2.imread("images.jpg")

# 그레이 스케일로 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 얼굴 감지
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
)

# 검출된 얼굴에 대해 박스 그리기 및 크기 출력
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    print(f"얼굴 크기: {w}x{h}")

# 이미지 출력
cv2.imshow("Faces Detected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
