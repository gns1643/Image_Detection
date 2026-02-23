import cv2
import numpy as np
import urllib.request
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ─── 모델 파일 자동 다운로드 ──────────────────────────────────────────────────
# MediaPipe 0.10+ 은 .tflite / .task 모델 파일을 직접 로드합니다.
# 파일이 없으면 Google 공식 저장소에서 자동으로 다운받습니다.

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(_MODEL_DIR, exist_ok=True)

_FACE_MODEL_PATH = os.path.join(_MODEL_DIR, "face_detection_short_range.tflite")
_POSE_MODEL_PATH = os.path.join(_MODEL_DIR, "pose_landmarker_lite.task")

_FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
_POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

def _download_if_missing(path: str, url: str) -> None:
    """모델 파일이 없으면 다운로드합니다."""
    if not os.path.exists(path):
        filename = os.path.basename(path)
        print(f"[모델 다운로드] {filename} 다운로드 중...")
        urllib.request.urlretrieve(url, path)
        print(f"[모델 다운로드] {filename} 완료 → {path}")

_download_if_missing(_FACE_MODEL_PATH, _FACE_MODEL_URL)
_download_if_missing(_POSE_MODEL_PATH, _POSE_MODEL_URL)


# ─── MediaPipe Tasks API 초기화 ───────────────────────────────────────────────
#
# [구버전 0.9 vs 신버전 0.10+ 차이점]
#
# 구버전:  mp.solutions.face_detection.FaceDetection(...)
#           → mp.Image 불필요, numpy 배열 직접 전달
#
# 신버전:  vision.FaceDetector.create_from_options(options)
#           → 반드시 mp.Image 로 변환 후 전달
#           → 모델 파일(.tflite / .task) 필요
#           → 결과 좌표가 "절대 픽셀값" 으로 반환 (구버전은 0~1 상대값)
#

# ─── 모델 파일을 Python이 직접 읽어 bytes로 전달 ─────────────────────────────
# main.py의 np.fromfile 방식과 동일한 원리:
# C++ 레이어가 한글 경로를 직접 열면 실패하므로,
# Python(유니코드 지원)이 먼저 읽고 bytes로 넘겨서 경로 문제를 우회합니다.
with open(_FACE_MODEL_PATH, 'rb') as f:
    _face_model_buffer = f.read()

with open(_POSE_MODEL_PATH, 'rb') as f:
    _pose_model_buffer = f.read()

# 얼굴 감지기 (단거리 모델: 웹캠 거리에 최적)
_face_options = vision.FaceDetectorOptions(
    base_options=mp_python.BaseOptions(model_asset_buffer=_face_model_buffer),
    min_detection_confidence=0.5
)
face_detector = vision.FaceDetector.create_from_options(_face_options)

# 포즈(상체) 감지기
_pose_options = vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_buffer=_pose_model_buffer),
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    num_poses=1
)
pose_detector = vision.PoseLandmarker.create_from_options(_pose_options)

# VIDEO 모드는 타임스탬프가 필요 → 프레임마다 증가시킬 카운터
_frame_timestamp_ms = 0


# ─── 상수 ────────────────────────────────────────────────────────────────────
A5_RATIO = 148 / 210  # ≈ 0.7048 (width / height)

# 상체 ROI 계산에 사용할 랜드마크 인덱스
# PoseLandmarker 랜드마크 목록: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
_UPPER_BODY_IDX = [0, 11, 12, 13, 14]
# 0=코, 11=왼어깨, 12=오른어깨, 13=왼팔꿈치, 14=오른팔꿈치 (골반 제외 → 중심 쏠림 방지)


# ─── 내부 유틸 ────────────────────────────────────────────────────────────────

def _to_mp_image(image: np.ndarray) -> mp.Image:
    """
    OpenCV BGR numpy 배열을 MediaPipe Image 로 변환합니다.
    신버전 Tasks API 는 mp.Image 만 입력으로 받습니다.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _adjust_roi_to_ratio(
    x1: int, y1: int, x2: int, y2: int,
    frame_h: int, frame_w: int,
    ratio: float = A5_RATIO
) -> tuple:
    """
    ROI를 지정 비율(width/height)로 중심 유지하며 조정합니다.
    프레임 경계를 벗어나지 않도록 클리핑합니다.
    """
    roi_w = x2 - x1
    roi_h = y2 - y1
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    current_ratio = roi_w / roi_h if roi_h > 0 else ratio

    if current_ratio > ratio:
        new_w = roi_w
        new_h = int(roi_w / ratio)
    else:
        new_h = roi_h
        new_w = int(roi_h * ratio)

    new_x1 = max(0,       cx - new_w // 2)
    new_y1 = max(0,       cy - new_h // 2)
    new_x2 = min(frame_w, new_x1 + new_w)
    new_y2 = min(frame_h, new_y1 + new_h)

    return new_x1, new_y1, new_x2, new_y2


# ─── 공개 함수 ────────────────────────────────────────────────────────────────

def detect_person_and_get_roi(image: np.ndarray) -> tuple | None:
    """
    MediaPipe PoseLandmarker로 상체 ROI를 반환합니다.

    [신버전 변경점]
    - VIDEO 모드: detect_for_video(mp_image, timestamp_ms) 사용
    - 랜드마크: result.pose_landmarks[사람인덱스][랜드마크인덱스]
    - 좌표: .x, .y 가 0~1 정규화값 → 픽셀 변환 필요
    - visibility: .visibility 로 접근 (구버전과 동일)

    Returns:
        (x, y, w, h) 또는 None
    """
    global _frame_timestamp_ms
    _frame_timestamp_ms += 33  # 약 30fps 기준 타임스탬프 증가

    h, w = image.shape[:2]
    result = pose_detector.detect_for_video(_to_mp_image(image), _frame_timestamp_ms)

    # 감지된 사람이 없으면 None
    if not result.pose_landmarks:
        return None

    # 첫 번째 사람(num_poses=1 이므로 항상 인덱스 0)
    landmarks = result.pose_landmarks[0]

    xs, ys = [], []
    for idx in _UPPER_BODY_IDX:
        lm = landmarks[idx]
        if lm.visibility > 0.3:
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))

    if len(xs) < 2:
        return None

    pad_x   = int((max(xs) - min(xs)) * 0.1)
    pad_y   = int((max(ys) - min(ys)) * 0.1)
    # 머리 위 여유 공간 확보: 위쪽 패딩을 아래쪽보다 크게
    pad_top = int((max(ys) - min(ys)) * 0.5)

    rx = max(0, min(xs) - pad_x)
    ry = max(0, min(ys) - pad_top)   # 위로 더 올림
    rw = min(w, max(xs) + pad_x) - rx
    rh = min(h, max(ys) + pad_y) - ry

    return (rx, ry, rw, rh)


def detect_face_and_get_roi(image: np.ndarray) -> tuple | None:
    """
    MediaPipe FaceDetector로 얼굴을 감지하고 A5 비율 ROI를 반환합니다.

    [신버전 변경점]
    - IMAGE 모드: detect(mp_image) 사용
    - 결과 좌표: bounding_box.origin_x/y, width, height → 절대 픽셀값
      (구버전은 0~1 상대값이었음 → 직접 곱셈 불필요)
    - confidence: detection.categories[0].score 로 접근

    Returns:
        A5 비율이 적용된 (x1, y1, x2, y2) 또는 None
    """
    h, w = image.shape[:2]
    result = face_detector.detect(_to_mp_image(image))

    if not result.detections:
        return None

    # confidence 가장 높은 얼굴 선택
    best = max(result.detections, key=lambda d: d.categories[0].score)
    bb = best.bounding_box

    # 신버전: bounding_box 는 이미 절대 픽셀값
    fx, fy   = bb.origin_x, bb.origin_y
    fw, fh   = bb.width,    bb.height

    # 얼굴 여백 추가 (위쪽=이마/머리, 아래쪽=턱 비대칭 패딩)
    pad_w        = int(fw * 0.2)
    pad_top      = int(fh * 0.5)   # 이마/머리 위 여유를 크게
    pad_bottom   = int(fh * 0.2)   # 턱 아래는 작게
    x1 = max(0, fx - pad_w)
    y1 = max(0, fy - pad_top)
    x2 = min(w, fx + fw + pad_w)
    y2 = min(h, fy + fh + pad_bottom)

    # A5 비율 조정
    x1, y1, x2, y2 = _adjust_roi_to_ratio(x1, y1, x2, y2, h, w)

    return (x1, y1, x2, y2)
