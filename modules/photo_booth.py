import cv2
import os
import time

def run_photo_booth():
    """
    사진 촬영 UI 실행
    - SPACE: 사진 촬영
    - ESC  : 종료
    반환값:
        촬영 성공 시 -> 저장된 이미지 경로 (str)
        종료 시      -> None
    """

    # 이 파일의 위치를 기준으로 프로젝트의 기본 디렉토리(루트)를 찾습니다. (두 단계 위)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 기존 프로젝트의 'data/raw' 폴더를 저장 위치로 설정합니다.
    SAVE_DIR = os.path.join(BASE_DIR, "data", "raw")
    os.makedirs(SAVE_DIR, exist_ok=True)

    def get_next_filename(folder, ext=".png"):
        files = [f for f in os.listdir(folder) if f.endswith(ext)]
        return os.path.join(folder, f"{len(files) + 1}{ext}")

    def draw_button(frame):
        h, w, _ = frame.shape
        cv2.rectangle(
            frame,
            (w // 2 - 120, h - 100),
            (w // 2 + 120, h - 40),
            (0, 255, 0),
            -1
        )
        cv2.putText(
            frame,
            "TAKE PHOTO (SPACE)",
            (w // 2 - 160, h - 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

    def countdown_and_capture(cap):
        for i in range(5, 0, -1):
            ret, frame = cap.read()
            if not ret:
                return None

            h, w, _ = frame.shape
            cv2.putText(
                frame,
                str(i),
                (w // 2 - 30, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                4,
                (0, 0, 255),
                6
            )

            cv2.imshow("Photo Booth", frame)
            cv2.waitKey(1000)

        ret, final_frame = cap.read()
        if not ret:
            return None

        save_path = get_next_filename(SAVE_DIR)
        cv2.imwrite(save_path, final_frame)
        print(f"📸 저장 완료: {save_path}")
        return save_path

    # ===============================
    # 메인 루프
    # ===============================
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ 카메라를 열 수 없습니다")

    print("▶ SPACE: 사진 촬영 | ESC: 종료")

    captured_path = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        draw_button(frame)
        cv2.imshow("Photo Booth", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE
            captured_path = countdown_and_capture(cap)
            break

        elif key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    return captured_path
