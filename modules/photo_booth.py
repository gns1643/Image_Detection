import cv2
import os
import time
import numpy as np

def run_photo_booth(save_dir=None):
    """
    사진 촬영 UI 실행
    - SPACE: 사진 촬영
    - ESC  : 종료
    Args:
        save_dir (str, optional): 이미지를 저장할 디렉토리. None이면 기본 경로 사용.
    반환값:
        촬영 성공 시 -> 저장된 이미지 경로 (str)
        종료 시      -> None
    """

    # 저장 경로 설정
    if save_dir:
        SAVE_DIR = save_dir
    else:
        # 이 파일의 위치를 기준으로 프로젝트의 기본 디렉토리(루트)를 찾습니다.
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        SAVE_DIR = os.path.join(BASE_DIR, "data", "raw")
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    def get_next_filename(ext=".png"):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}{ext}"

    def draw_button(frame):
        h, w, _ = frame.shape
        # UI 요소의 크기와 위치를 프레임 크기에 비례하여 조정할 수 있습니다.
        # 예: cv2.rectangle(frame, (int(w*0.4), int(h*0.85)), ...)
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
            (w // 2 - 100, h - 65), # 텍스트 위치 조정
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

    def countdown_and_capture(cap):
        for i in range(3, 0, -1): # 카운트다운 5초 -> 3초
            # 매 카운트마다 새로운 프레임을 읽어 화면에 표시
            for _ in range(30): # 1초간 여러 프레임을 보여줘서 부드럽게
                ret, frame = cap.read()
                if not ret: return None
                
                # 좌우 반전 추가
                frame = cv2.flip(frame, 1)
                
                h, w, _ = frame.shape
                cv2.putText(
                    frame,
                    str(i),
                    (w // 2 - 40, h // 2 + 40), # 숫자 위치 조정
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4,
                    (0, 0, 255),
                    6
                )
                cv2.imshow("Photo Booth", frame)
                cv2.waitKey(33) # 약 30fps

        ret, final_frame = cap.read()
        if not ret:
            return None
        
        # 최종 프레임도 좌우 반전
        final_frame = cv2.flip(final_frame, 1)

        filename = get_next_filename()
        save_path = os.path.join(SAVE_DIR, filename)

        # 한글 경로 지원을 위해 imencode와 tofile 사용
        result, encoded_img = cv2.imencode(".png", final_frame)
        if result:
            encoded_img.tofile(save_path)
            print(f"📸 저장 완료: {save_path}")
            return save_path
        else:
            print("❌ 이미지 인코딩에 실패했습니다.")
            return None

    # ===============================
    # 메인 루프
    # ===============================
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다. 다른 프로그램이 카메라를 사용 중인지 확인하세요.")
        return None # 예외 대신 None 반환

    print("▶ SPACE: 사진 촬영 | ESC: 종료")
    captured_path = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 카메라에서 프레임을 읽을 수 없습니다.")
            break

        # 좌우 반전 추가
        frame = cv2.flip(frame, 1)

        draw_button(frame)
        cv2.imshow("Photo Booth", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE
            captured_path = countdown_and_capture(cap)
            break
        elif key == 27:  # ESC
            print(">> 포토부스를 종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()

    return captured_path