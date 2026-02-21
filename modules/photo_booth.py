import cv2
import os
import time
import numpy as np

from modules.human_cropper import detect_person_and_get_roi, detect_face_and_get_roi

# ─── 상수 ────────────────────────────────────────────────────────────────────
BUTTON_HALF_WIDTH   = 120
BUTTON_HEIGHT_OFFSET_TOP    = 100
BUTTON_HEIGHT_OFFSET_BOTTOM = 40
TARGET_FPS = 30
WAIT_MS    = int(1000 / TARGET_FPS)  # 33ms


# ─── 유틸 함수 ────────────────────────────────────────────────────────────────

def get_next_filename(ext: str = ".png") -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}{ext}"


def draw_button(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(
        frame,
        (w // 2 - BUTTON_HALF_WIDTH,  h - BUTTON_HEIGHT_OFFSET_TOP),
        (w // 2 + BUTTON_HALF_WIDTH,  h - BUTTON_HEIGHT_OFFSET_BOTTOM),
        (0, 255, 0),
        -1
    )
    cv2.putText(
        frame,
        "TAKE PHOTO (SPACE)",
        (w // 2 - 100, h - 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )


def countdown_and_capture(cap: cv2.VideoCapture, save_dir: str) -> str | None:
    """카운트다운 후 촬영하여 저장 경로 반환."""
    for i in range(3, 0, -1):
        start = time.time()
        while time.time() - start < 1.0:   # 정확히 1초 대기
            ret, frame = cap.read()
            if not ret:
                return None
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            cv2.putText(
                frame, str(i),
                (w // 2 - 40, h // 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                4, (0, 0, 255), 6
            )
            cv2.imshow("Photo Booth", frame)
            cv2.waitKey(WAIT_MS)

    ret, final_frame = cap.read()
    if not ret:
        return None

    final_frame = cv2.flip(final_frame, 1)

    filename  = get_next_filename()
    save_path = os.path.join(save_dir, filename)

    result, encoded_img = cv2.imencode(".png", final_frame)
    if result:
        encoded_img.tofile(save_path)
        print(f"📸 저장 완료: {save_path}")
        return save_path

    print("❌ 이미지 인코딩에 실패했습니다.")
    return None


# ─── 메인 함수 ───────────────────────────────────────────────────────────────

def run_photo_booth(save_dir: str = None) -> str | None:
    """
    사진 촬영 UI 실행
    - SPACE: 사진 촬영
    - ESC  : 종료

    Args:
        save_dir: 이미지를 저장할 디렉토리. None이면 기본 경로 사용.

    Returns:
        촬영 성공 시 저장된 이미지 경로, 종료 시 None
    """
    if save_dir is None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(BASE_DIR, "data", "raw")

    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return None

    print("▶ SPACE: 사진 촬영 | ESC: 종료")
    captured_path = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 카메라에서 프레임을 읽을 수 없습니다.")
                break

            frame = cv2.flip(frame, 1)

            # 사람(상체) 감지
            person_roi = detect_person_and_get_roi(frame)
            if person_roi:
                px, py, pw, ph = person_roi
                cv2.rectangle(frame, (px, py), (px+pw, py+ph), (0, 255, 0), 2)

                # 상체 영역에서 얼굴 감지
                person_frame = frame[py:py+ph, px:px+pw]
                face_roi = detect_face_and_get_roi(person_frame)

                if face_roi:
                    fx1, fy1, fx2, fy2 = face_roi

                    # 원본 프레임 기준 절대 좌표로 변환
                    abs_x1 = px + fx1
                    abs_y1 = py + fy1
                    abs_x2 = px + fx2
                    abs_y2 = py + fy2

                    # A5 비율 조정은 human_cropper 내부에서 완료됨
                    cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (255, 0, 0), 2)
                    cv2.putText(
                        frame, "A5",
                        (abs_x1 + 4, abs_y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 0), 2
                    )

            draw_button(frame)
            cv2.imshow("Photo Booth", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:   # SPACE
                captured_path = countdown_and_capture(cap, save_dir)
                break
            elif key == 27: # ESC
                print(">> 포토부스를 종료합니다.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return captured_path