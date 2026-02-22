import os
import datetime

# --- 기본 경로 설정 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
# 사용자가 이미지를 넣는 공용 폴더
GENERAL_INPUT_DIR = os.path.join(DATA_DIR, "raw") 

# --- 세션 관리 ---
# 프로그램의 단일 실행 인스턴스 동안 유지되는 최상위 세션 폴더 경로
SESSION_DIR = None

def initialize_session():
    """
    프로그램 실행 시 단 한 번만 호출되어 메인 세션 폴더를 생성합니다.
    이 함수는 생성된 최상위 세션 폴더의 경로를 반환합니다.
    """
    global SESSION_DIR
    if SESSION_DIR is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_folder_name = f"{timestamp}_session"
        SESSION_DIR = os.path.join(DATA_DIR, session_folder_name)
        os.makedirs(SESSION_DIR, exist_ok=True)
        print(f"[세션 초기화] 메인 세션 폴더가 생성되었습니다: {SESSION_DIR}")
    return SESSION_DIR

def setup_photo_paths(photo_name="photo"):
    """
    메인 세션 폴더 아래에 개별 사진 처리를 위한 하위 폴더
    (raw, intermediate, processed)를 생성하고 각 경로를 반환합니다.
    """
    if SESSION_DIR is None:
        raise RuntimeError("세션이 초기화되지 않았습니다. initialize_session()을 먼저 호출해야 합니다.")

    photo_folder_path = os.path.join(SESSION_DIR, photo_name)

    # 개별 사진 폴더 내의 하위 폴더 경로 설정
    input_dir = os.path.join(photo_folder_path, "raw")
    intermediate_dir = os.path.join(photo_folder_path, "intermediate")
    output_dir = os.path.join(photo_folder_path, "processed")

    # 모든 폴더 생성
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"  - 사진 처리 폴더를 생성했습니다: {photo_folder_path}")
    return input_dir, intermediate_dir, output_dir

# --- 처리 모드 선택 ---
PROCESSING_MODE_CROP_PERSON_FACE = "PERSON_AND_FACE" # 사람 감지 -> 얼굴 감지
PROCESSING_MODE_CROP_FACE_ONLY = "FACE_ONLY"         # 얼굴만 바로 감지
PROCESSING_MODE_NO_CROP = "NO_CROP"                  # 자르기 없음 (전체 이미지)

# --- 외곽선 검출 알고리즘 선택 ---
EDGE_DETECTION_CANNY = "CANNY"
EDGE_DETECTION_DEXINED = "DEXINED"

# --- 알고리즘별 설정값 ---
# Canny 엣지 설정
FINAL_TH1 = 50
FINAL_TH2 = 100
# DexiNed 이진화 임계값
DEXINED_THRESHOLD = 128

# G-코드(로봇) 설정
Z_SAFE = 10.0       # 이동 높이 (mm)
Z_DRAW = 0.0        # 그리기 높이 (mm)
FEED_RATE = 1000    # 속도 (mm/min)
SCALE = 0.2         # 크기 조절 (1픽셀 = 0.2mm)
