import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import utils, math
import winsound

frame_counter =0
CEF_COUNTER =0
TOTAL_BLINKS =0
CLOSED_EYES_FRAME =3
FONTS =cv.FONT_HERSHEY_COMPLEX
# 왼쪽 눈 렌드마크
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# 오른쪽 눈 렌드마크
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  

map_face_mesh = mp.solutions.face_mesh
camera = cv.VideoCapture(0)

def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    return mesh_coord

def euclaideanDistance(point1, point2):
    x, y = point1
    x1, y1 = point2
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

def blinkRatio(img, landmarks, right_indices, left_indices):
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]
    
    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)
    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance
    ratio = (reRatio+leRatio)/2
    return ratio 

with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:
    start_time = time.time()
    # 비디오 시작
    while True:
        frame_counter +=1 
        ret, frame = camera.read() 
        if not ret: 
            break
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width= frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results  = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
            # cv.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, utils.GREEN, 2)
            utils.colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)


            if ratio >3.5:
                CEF_COUNTER +=1
                if CEF_COUNTER >= 50:
                    utils.colorBackgroundText(frame,  f'wake up', FONTS, 3, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )
                    winsound.Beep(frequency=800,duration=350)
            if ratio < 3.5:
                a = 100
                CEF_COUNTER = 0
            else:
                pass
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)

        end_time = time.time()-start_time
        fps = frame_counter/end_time

        frame =utils.textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
        cv.imshow('frame',frame)
        key = cv.waitKey(2)
        if key==ord('q') or key ==ord('Q'):
            break
    cv.destroyAllWindows()
    camera.release()