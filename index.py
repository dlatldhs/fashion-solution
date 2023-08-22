import lib_import
import functions

# mediapipe_define
mp_drawing = lib_import.mp.solutions.drawing_utils
mp_drawing_styles = lib_import.mp.solutions.drawing_styles
mp_pose = lib_import.mp.solutions.pose

# mediapipe pose model load
pose = mp_pose.Pose(
    min_detection_confidence=0.5, # 기본값
    min_tracking_confidence=0.5,  # 기본값
    model_complexity=2  # 모델 복잡도 -> 정확도 증가
)

# img read
img_path = './images.jpg'
img = lib_import.cv2.imread(img_path)

# img variable
img_h, img_w, _ = img.shape
original_img = img.copy()
mtcnn_img = img.copy()
# drawing variable
color = (0,255,0) # green
thickness = 2
radius = 5

# img brg -> rgb transform
img = lib_import.cv2.cvtColor(img,lib_import.cv2.COLOR_BGR2RGB)

# pose
pose_locate = pose.process(img)

# 관절 위치 그리기
mp_drawing.draw_landmarks(
    original_img,
    pose_locate.pose_landmarks,
    mp_pose.POSE_CONNECTIONS,
    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
)

# LandMark drawing
if pose_locate.pose_landmarks:

    # landmark list 
    landmark_list = pose_locate.pose_landmarks.landmark

    # find shoulder locate
    left_shoulder_x = landmark_list[mp_pose.PoseLandmark.LEFT_SHOULDER].x * img_w
    right_shoulder_x = landmark_list[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * img_w

    # get shoulder locate
    shoulder_length = left_shoulder_x - right_shoulder_x

    # find hip locate
    l_hip_x = landmark_list[mp_pose.PoseLandmark.LEFT_HIP].x * img_w
    r_hip_x = landmark_list[mp_pose.PoseLandmark.RIGHT_HIP].x * img_w

    # get hip locate
    hip_length = l_hip_x - r_hip_x

    #  shoulder compare hip
    shoulder_hip_diff, shoulder_result = functions.get_shoulder_len(shoulder_length,hip_length)

    lib_import.cv2.imshow("original img",original_img)
    lib_import.cv2.waitKey(0)

pose.close()

print(shoulder_hip_diff, shoulder_result)