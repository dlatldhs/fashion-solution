import lib_import

def get_shoulder_len( shoulder, hip ):
    
    sh_diff = shoulder - hip
    
    if sh_diff < 5 :
        return -1
    
    else :
        if sh_diff > 0 and sh_diff < 41:
            result = 4
            print('bad')
        elif sh_diff > 40 and sh_diff < 61:
            result = 3
            print('normal')
        elif sh_diff > 60 and sh_diff < 81:
            result = 2
            print('good')
        else:
            result = 1
            print('very good')
    
    print(sh_diff)

    return sh_diff,result

def get_face_size( img ):
    
    image = img
    # mtcnn model initialization
    mtcnn = lib_import.MTCNN()

    # color img -> rgb img transform
    rgb_img = lib_import.cv2.cvtColor(image, lib_import.cv2.COLOR_BGR2RGB)

    # face detect
    faces = mtcnn.detect_faces(rgb_img)

    # drawing box and printing
    for face in faces:
        x, y, w, h = face["box"]
        lib_import.cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        print(f"face size: {w}x{h}")
    
    return image, (w,h)

def ratio_solution( img ):

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
        shoulder_hip_diff, shoulder_result = get_shoulder_len(shoulder_length,hip_length)

        # img check
        # lib_import.cv2.imshow("original img",original_img)
        
        face_detecting_img, (face_w,face_h) = get_face_size(mtcnn_img)

        body_ratio = round(body_length/face_h,2)

        # img base64 encoding
        _, buffer = lib_import.cv2.imencode('.jpg', face_detecting_img)
        encoded_image = lib_import.base64.b64encode(buffer).decode('utf-8')

        # debuging img
        lib_import.cv2.imshow("face detecting img", face_detecting_img)
        lib_import.cv2.imshow("test",original_img)
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

    result_list = [shoulder_hip_diff,shoulder_result,face_w,face_h,body_length,body_ratio]
    
    return encoded_image,result_list