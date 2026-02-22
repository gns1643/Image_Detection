import cv2
import numpy as np
import os

# OpenCV에 내장된 Haar Cascade 모델 파일 경로를 찾습니다.
# 이 모델은 사람의 얼굴과 상체를 감지하는 데 사용됩니다.
try:
    cascades_path = cv2.data.haarcascades
    if not os.path.isdir(cascades_path):
        raise FileNotFoundError
except Exception as e:
    print(f"[오류] OpenCV의 Haar Cascade 데이터 경로를 찾을 수 없습니다: {e}")
    # 대체 경로를 시도하거나 사용자에게 경로 설정을 요청할 수 있습니다.
    # 여기서는 예외적으로 많이 사용되는 경로 중 하나를 지정합니다.
    # 실제 환경에 따라 이 경로는 수정이 필요할 수 있습니다.
    cascades_path = os.path.join(os.path.dirname(cv2.__file__), 'data')


FACE_CASCADE_PATH = os.path.join(cascades_path, 'haarcascade_frontalface_default.xml')
BODY_CASCADE_PATH = os.path.join(cascades_path, 'haarcascade_upperbody.xml') # 상체 감지가 더 효과적일 수 있습니다.

# 모델 파일 존재 여부 확인
if not os.path.exists(FACE_CASCADE_PATH) or not os.path.exists(BODY_CASCADE_PATH):
    print("[오류] 오류: 사람 또는 얼굴 감지를 위한 모델 파일(haarcascade)이 없습니다.")
    print(f"   - 얼굴 모델 경로: {FACE_CASCADE_PATH}")
    print(f"   - 신체 모델 경로: {BODY_CASCADE_PATH}")
    # 필요하다면 여기서 스크립트를 중단하거나, 사용자에게 파일을 다운로드하라고 안내할 수 있습니다.
    face_cascade = None
    body_cascade = None
else:
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    body_cascade = cv2.CascadeClassifier(BODY_CASCADE_PATH)

def detect_and_crop_person(image: np.ndarray) -> np.ndarray:
    """
    이미지에서 가장 큰 사람(상체)을 감지하고 해당 부분을 잘라냅니다.
    
    :param image: 원본 이미지 (OpenCV 형식)
    :return: 사람이 감지되면 잘라낸 이미지, 아니면 None을 반환합니다.
    """
    if body_cascade is None:
        print("[경고] 경고: 신체 감지 모델이 로드되지 않아 사람 감지 단계를 건너뜁니다.")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bodies = body_cascade.detectMultiScale(gray, 1.1, 4)

    if len(bodies) == 0:
        print("[정보] 정보: 이미지에서 사람(상체)을 찾지 못했습니다.")
        return None # 사람을 못찾으면 None 반환

    # 감지된 사람 중 가장 큰 영역을 찾습니다 (너비*높이 기준).
    largest_body = max(bodies, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_body
    
    print(f">> 사람(상체) 감지 완료. 위치: (x={x}, y={y}, w={w}, h={h})")
    
    # 감지된 영역을 잘라냅니다.
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def detect_and_crop_face(image: np.ndarray) -> np.ndarray:
    """
    이미지에서 가장 큰 얼굴을 감지하고 해당 부분을 잘라냅니다.
    
    :param image: 사람만 잘라낸 이미지 또는 원본 이미지
    :return: 얼굴이 감지되면 잘라낸 이미지, 아니면 None을 반환합니다.
    """
    if face_cascade is None:
        print("[경고] 경고: 얼굴 감지 모델이 로드되지 않아 얼굴 감지 단계를 건너뜁니다.")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        print("[정보] 정보: 이미지에서 얼굴을 찾지 못했습니다.")
        return None # 얼굴을 못찾으면 None 반환

    # 가장 큰 얼굴을 선택합니다.
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face

    print(f">> 얼굴 감지 완료. 위치: (x={x}, y={y}, w={w}, h={h})")

    # 잘라낼 영역에 여백(padding)을 추가합니다.
    padding_w = int(w * 0.2)
    padding_h = int(h * 0.4)
    
    # 여백을 적용한 새로운 좌표를 계산합니다.
    y1 = max(0, y - padding_h)
    y2 = min(image.shape[0], y + h + padding_h)
    x1 = max(0, x - padding_w)
    x2 = min(image.shape[1], x + w + padding_w)

    print(f"   - 여백 적용 후, 자를 영역: (y1={y1}, y2={y2}, x1={x1}, x2={x2})")

    # 얼굴 부분을 여유있게 잘라냅니다.
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image
