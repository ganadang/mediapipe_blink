# -3502 김민섭-
동신AI 대회

## 사용한 주요 모듈
```p
import cv2 
import numpy 
import mediapipe
import utils

```

## 주요 코드 설명

### 눈 주위의 렌드마크
 ```c
# 왼쪽 눈 렌드마크
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# 오른쪽 눈 렌드마크
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
```
눈 주위의 필요한 렌드마크를 가져온다
```p
def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]
    return mesh_coord
```
불러온 렌드마크를 지정한다.

### 깜박임 판단하기
#### 유클리드거리 지정하기
```p
def euclaideanDistance(point1, point2):
    x, y = point1
    x1, y1 = point2
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance
```
눈의 가로와 세로

#### 화면에서 눈 일고 비율 측정하기
```p
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
```
렌드마크에서 왼쪽, 오른쪽눈의 렌드마크에서 세로의 최대,최소 가로의 최대,최소지점을 받는다.
받은 정보로 유클리드 거리를 구한후 왼쪽눈의 가로세로 비율,오른쪽눈의 가로세로 비율의 평균을 전체 평균으로 가진다.
