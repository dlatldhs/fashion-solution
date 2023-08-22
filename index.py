import lib_import
import functions

# flask app
app = lib_import.Flask(__name__)

# cors
lib_import.CORS(
        app,
        resources={r"/api/*": {"origins":"*"}},
        supports_credentials=True
    )

@app.route('/')
def main_page():
    return lib_import.render_template('index.html')

@app.route("/body_ratio", methods=['POST'] )
def body_ratio_survey():

    file = lib_import.request.files['file']
    filestr = file.read()
    npimg = lib_import.np.fromstring(filestr, lib_import.np.uint8)
    img = lib_import.cv2.imdecode(npimg, lib_import.cv2.IMREAD_COLOR)

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
    # img_path = './images.jpg'
    # img = lib_import.cv2.imread(img_path)

    # img variable
    img_h, img_w, _ = img.shape
    original_img = img.copy()
    mtcnn_img = img.copy()
    
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
        
        left_shoulder_y = landmark_list[mp_pose.PoseLandmark.LEFT_SHOULDER].y * img_h
        right_shoulder_y = landmark_list[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * img_h

        sum_shoulder = (left_shoulder_y + right_shoulder_y)/2

        # get shoulder locate
        shoulder_length = left_shoulder_x - right_shoulder_x

        # find hip locate
        l_hip_x = landmark_list[mp_pose.PoseLandmark.LEFT_HIP].x * img_w
        r_hip_x = landmark_list[mp_pose.PoseLandmark.RIGHT_HIP].x * img_w

        # find foot locate
        
        l_foot_y = landmark_list[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * img_h
        r_foot_y = landmark_list[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * img_h

        sum_foot = (l_foot_y+r_foot_y)/2
        
        # all body length = foot_sum_locate_y + sholder_sum_locate_y
        body_length = sum_foot-sum_shoulder

        # get hip locate
        hip_length = l_hip_x - r_hip_x

        #  shoulder compare hip
        shoulder_hip_diff, shoulder_result = functions.get_shoulder_len(shoulder_length,hip_length)

        # img check
        lib_import.cv2.imshow("original img",original_img)
        
        face_detecting_img, (face_w,face_h) = functions.get_face_size(mtcnn_img)

        body_ratio = round(body_length/face_h,2)

        # debuging img
        lib_import.cv2.imshow("face detecting img", face_detecting_img)
        lib_import.cv2.waitKey(0)
        

    pose.close()

    body_info = {
        "shoulder_rate" : shoulder_result,
        "shoulder_hip_diff" : shoulder_hip_diff,
        "face_width" : face_w,
        "face_height": face_h,
        "body_length": body_length,
        "body_ratio" : body_ratio, 
    }

    print(f"어깨 힙 차이: {shoulder_hip_diff}, 어깨 등급 : {shoulder_result}")
    print(f"얼굴 크기, w :{face_w} , h :{face_h}")
    print(f"몸길이 {body_length} {body_ratio}등신 입니다")

    result = lib_import.jsonify(body_info)
    return result
    
def main():
    app.debug = True
    app.run(host="localhost", port="8080")

if __name__ == "__main__":
    main()