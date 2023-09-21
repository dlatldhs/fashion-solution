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

img = 0

@app.route('/')
def main_page():
    return lib_import.render_template('main.html')

@app.route('/capture')
def capture_page():
    
    # 카메라에 접근
    cap = lib_import.cv2.VideoCapture(lib_import.cv2.CAP_DSHOW+1)

    # 비디오 스트림에서 프레임을 읽어옴
    ret, frame = cap.read()
    global img
    img = frame
    # lib_import.cv2.imshow("ret",frame)
    _, buffer = lib_import.cv2.imencode('.jpg', frame)
    img = lib_import.base64.b64encode(buffer).decode('utf-8')
    
    # 카메라 해제
    cap.release()
    return lib_import.render_template('capture.html',img=img)

@app.route("/file_sumit")
def get_picture_file():
    return lib_import.render_template('index.html')

@app.route("/body_ratio", methods=['POST'] )
def body_ratio_survey():
    file = lib_import.request.files['file']
    filestr = file.read()
    npimg = lib_import.np.fromstring(filestr, lib_import.np.uint8)
    img = lib_import.cv2.imdecode(npimg, lib_import.cv2.IMREAD_COLOR)
    image ,points_image, result_list = functions.ratio_solution(img)

    return lib_import.render_template(
        'result.html',
        image_data=image,
        points_image=points_image,
        shoulder_hip_diff=result_list[0],
        shoulder_result=result_list[1],
        face_w=result_list[2],
        face_h=result_list[3],
        body_length=result_list[4],
        body_ratio=result_list[5]
    )
    
@app.route("/body_ratio2", methods=['POST'] )
def body_ratio_survey2():
    global img
    decoded_data = lib_import.base64.b64decode(img)
    np_data = lib_import.np.frombuffer(decoded_data, lib_import.np.uint8)
    
    image = lib_import.cv2.imdecode(np_data, lib_import.cv2.IMREAD_ANYCOLOR)
    image, points_image ,result_list = functions.ratio_solution(image)

    return lib_import.render_template(
        'result.html',
        image_data=image,
        points_image=points_image,
        shoulder_hip_diff=result_list[0],
        shoulder_result=result_list[1],
        face_w=result_list[2],
        face_h=result_list[3],
        body_length=result_list[4],
        body_ratio=result_list[5]
    )

def main():
    app.debug = True
    app.run(host="localhost", port="8080")

if __name__ == "__main__":
    main()