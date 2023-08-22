import lib_import

def get_shoulder_len( shoulder, hip ):
    
    sh_diff = shoulder - hip
    
    if sh_diff < 0 :
        return -1
    
    else :
        if sh_diff > 0 and sh_diff < 41:
            result = 1
            print('bad')
        elif sh_diff > 40 and sh_diff < 61:
            result = 2
            print('normal')
        elif sh_diff > 60 and sh_diff < 81:
            result = 3
            print('good')
        else:
            result = 4
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