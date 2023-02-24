# - 김민섭- 
탐구보고서
## 사용한 주요 모듈
```c
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
```c
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
```c
def euclaideanDistance(point1, point2):
    x, y = point1
    x1, y1 = point2
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance
```
눈의 가로와 세로

#### 화면에서 눈 일고 비율 측정하기
```c
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
렌드마크에서 왼쪽, 오른쪽눈의 세로의 최대,최소 가로의 최대,최소지점을 받는다.
받은 정보로 유클리드 거리를 구한후 구한 정보를 바탕으로 왼쪽눈의 가로세로 비율,오른쪽눈의 가로세로 비율의 평균을 전체 평균으로 가진다.

## 유클리드 거리 비율을 사용한 이유

처음에 눈의 크기를 측정하기위에 단순히 세로의 길이로만 판단하고, 유클리드를 사용하기도했다. 
하지만 달리는 차 안에서 머리가 항상 같은 위치에 고정되어있을 수는 없다는 큰 문제점이 있었다.
길이, 유클리드로 판단하는 방법은 머리의 앞,뒤,기울어진 위치에 따라 계산 값을 계속 바꿔주어야 했기 때문이다.
유클리드 거리 비율을 사용하면 머리의 위치와 관계없이 항상 같은 정보를 받을 수 있었다.

## mediapipe
MediaPipe란 구글에서 제공하는 AI 프레임워크로서, 비디오형식 데이터를 이용한 다양한 비전 AI 기능을 파이프라인 형태로 손쉽게 사용할 수 있도록 제공된다.  AI 모델개발 및 수많은 데이터셋을 이용한 학습도 마친 상태로 제공되므로 라이브러브 불러 사용하듯이 간편하게 호출하여 사용하기만 하면 되는 형태로 비전 AI 기능을 개발할 수 있다.  
![mediapipe_small](https://user-images.githubusercontent.com/116184495/196741353-b46bab77-86b7-4cd2-9dfe-7e44ce401e4a.png)
[<링크>](https://google.github.io/mediapipe/)

### mediapipe iris
mediapipe에서 iris기술을 사용하였다.
mediapipe iris는 홍채를 추적하여 눈 위치를 추정하는 기술이다.
[<링크>](https://google.github.io/mediapipe/solutions/iris)

### mediapipe face mesh
![mediapipe_face_mesh](https://user-images.githubusercontent.com/116184495/196740116-4a099088-ea61-47cd-98c8-17e5a79bfb0c.jpg)

MediaPipe face mesh란 얼굴 그물망은 실시간으로 468개의 3D 얼굴 랜드마크를 추정하는 얼굴 형상 솔루션이다.
머신러닝 기술을 사용하여 3D표면 현상을 유추하로 전용 센서 없이 일반 카메라만으로 사용이 가능하다.
mediapipe face mesh 에서 눈 렌드마크를 찾았다.

