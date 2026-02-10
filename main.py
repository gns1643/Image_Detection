import sys
import os
import cv2
import numpy as np
from modules import (
    config, 
    preprocess_image, 
    generate_files_canny, 
    detect_and_crop_person, 
    detect_and_crop_face,
    run_photo_booth  # [수정] 모듈 추가
)

def main():
    print(">> 심플 G-코드 변환기 (얼굴 인식 포함) 시작")

    while True:
        # --- [수정] 입력 방식 선택 ---
        input_path = None
        
        print("\n[작업 모드 선택]")
        print("1. 웹캠으로 사진 촬영하기")
        print("2. 기존 이미지 파일 불러오기")
        print("Q. 프로그램 종료")
        mode = input("번호를 입력하세요 (1, 2 또는 Q): ").strip().upper()

        if mode == 'Q':
            print("프로그램을 종료합니다.")
            break

        if mode == '1':
            print("\n>> 포토부스를 실행합니다. (스페이스바: 촬영, ESC: 취소)")
            # run_photo_booth는 촬영된 파일의 전체 경로를 반환합니다.
            captured_path = run_photo_booth()
            
            if captured_path is None:
                print("[알림] 촬영이 취소되었습니다.")
                continue
            
            input_path = captured_path
            print(f">> 촬영 완료! 이미지 경로: {input_path}")

        elif mode == '2':
            filename = input(f"'{config.INPUT_DIR}' 에 있는 이미지 파일명을 입력하세요: ").strip().strip('"')
            input_path = os.path.join(config.INPUT_DIR, filename)
        
        else:
            print("[오류] 잘못된 입력입니다.")
            continue

        # --- 1단계: 이미지 로드 및 확인 ---
        print(f"\n[1단계] 이미지 로드: {input_path}")
        if not os.path.exists(input_path):
            print(f"[오류] 파일이 없습니다: {input_path}")
            continue

        # 한글 경로 처리 등을 위해 numpy로 읽어서 디코딩
        try:
            img_array = np.fromfile(input_path, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[오류] 파일을 읽는 중 에러 발생: {e}")
            continue

        if image is None:
            print(f"[오류] 이미지를 불러올 수 없습니다. 파일이 손상되었거나 이미지 파일이 아닐 수 있습니다.")
            continue

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        
        # --- 2단계: 사람 및 얼굴 감지/자르기 ---
        person_image = detect_and_crop_person(image)
        if person_image is None:
            print("[오류] 사람을 감지하지 못했습니다.")
            continue

        # --- 중간 과정 저장 ---
        intermediate_dir = 'data/intermediate'
        os.makedirs(intermediate_dir, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(input_path))[0]

        # 1. 잘라낸 사람 저장
        person_intermediate_filename = f"{base_filename}_cropped_person.jpg"
        person_intermediate_path = os.path.join(intermediate_dir, person_intermediate_filename)
        is_success_person, im_buf_arr_person = cv2.imencode(".jpg", person_image)
        if is_success_person:
            im_buf_arr_person.tofile(person_intermediate_path)
            print(f"중간 저장: 잘라낸 사람 이미지를 '{person_intermediate_path}'에 저장했습니다.")

        face_image = detect_and_crop_face(person_image)
        if face_image is None:
            print("[오류] 얼굴을 감지하지 못했습니다.")
            continue

        # 2. 잘라낸 얼굴 저장
        face_intermediate_filename = f"{base_filename}_cropped_face.jpg"
        face_intermediate_path = os.path.join(intermediate_dir, face_intermediate_filename)
        is_success_face, im_buf_arr_face = cv2.imencode(".jpg", face_image)
        if is_success_face:
            im_buf_arr_face.tofile(face_intermediate_path)
            print(f"중간 저장: 잘라낸 얼굴 이미지를 '{face_intermediate_path}'에 저장했습니다.")

        # --- 3단계: 이미지 전처리 (배경 제거 + 블러) ---
        img_blur = preprocess_image(face_image)
        if img_blur is None: 
            print("[오류] 이미지 전처리 중 문제가 발생했습니다.")
            continue
        
        # --- 4단계: 변환 및 저장 (Canny 방식) ---
        output_base_name = f"{base_filename}_face"
        nc_path = os.path.join(config.OUTPUT_DIR, f"{output_base_name}.nc")
        svg_path = os.path.join(config.OUTPUT_DIR, f"{output_base_name}.svg")
        
        generate_files_canny(img_blur, nc_path, svg_path)
        
        print(f"\n모든 작업 완료! '{output_base_name}' 이름으로 파일이 저장되었습니다.")
        print("-" * 30)

if __name__ == "__main__":
    main()