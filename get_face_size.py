import cv2
from mtcnn import MTCNN

# MTCNN 모델 초기화
mtcnn = MTCNN()

# 이미지 읽기
img = cv2.imread("images.jpg")

# 컬러 이미지를 RGB로 변환
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 얼굴 검출
faces = mtcnn.detect_faces(rgb_img)

# 검출된 얼굴에 대해 박스 그리기 및 크기 출력
for face in faces:
    x, y, w, h = face["box"]
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    print(f"얼굴 크기: {w}x{h}")

# 이미지 출력
cv2.imshow("Faces Detected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()