import sys
import os
import cv2
import numpy as np
import shutil
from modules import (
    config,
    preprocess_image,
    generate_files_canny,
    detect_and_crop_person,
    detect_and_crop_face,
    run_photo_booth
)

def main():
    print(">> 심플 G-코드 변환기 (얼굴 인식 포함) 시작")

    # 애플리케이션 시작 시 사용자가 이미지를 넣을 공용 폴더 생성
    os.makedirs(config.GENERAL_INPUT_DIR, exist_ok=True)
    
    # 프로그램 실행 시 단 한번, 메인 세션 폴더를 초기화
    config.initialize_session()
    photo_counter = 1

    while True:
        # 경로 변수 초기화
        input_path = None
        
        print("\n[작업 모드 선택]")
        print("1. 웹캠으로 사진 촬영하기")
        print("2. 기존 이미지 파일 불러오기")
        print("Q. 프로그램 종료")
        mode = input(f"번호를 입력하세요 (1, 2 또는 Q): ").strip().upper()

        if mode == 'Q':
            print("프로그램을 종료합니다.")
            break

        base_photo_name = None
        photo_specific_name = None

        if mode == '1':
            base_photo_name = "webcam"
            photo_specific_name = f"{photo_counter}_{base_photo_name}"
            
            # 개별 사진 처리를 위한 폴더들(raw, inter, proc) 생성
            input_dir, intermediate_dir, output_dir = config.setup_photo_paths(photo_specific_name)
            
            print("\n>> 포토부스를 실행합니다. (스페이스바: 촬영, ESC: 취소)")
            captured_path = run_photo_booth(save_dir=input_dir)
            
            if captured_path is None:
                print("[알림] 촬영이 취소되었습니다.")
                # photo_counter를 증가시키지 않고 루프의 처음으로 돌아감
                continue
            
            input_path = captured_path
            # 저장된 파일명을 기반으로 최종 파일명 결정
            base_filename = os.path.splitext(os.path.basename(input_path))[0]

        elif mode == '2':
            print(f"\n'{config.GENERAL_INPUT_DIR}' 폴더의 이미지 목록:")
            try:
                files = [f for f in os.listdir(config.GENERAL_INPUT_DIR) if os.path.isfile(os.path.join(config.GENERAL_INPUT_DIR, f))]
                if not files:
                    print("- 이미지가 없습니다. 해당 폴더에 이미지를 추가해주세요.")
                    continue
                for f in files:
                    print(f"- {f}")
            except FileNotFoundError:
                print(f"- 폴더가 없습니다: '{config.GENERAL_INPUT_DIR}'")
                continue

            filename = input("파일명을 입력하세요: ").strip().strip('"')
            original_input_path = os.path.join(config.GENERAL_INPUT_DIR, filename)

            if not os.path.exists(original_input_path):
                print(f"[오류] 파일이 없습니다: {original_input_path}")
                continue
            
            base_filename = os.path.splitext(filename)[0]
            photo_specific_name = f"{photo_counter}_{base_filename}"

            # 개별 사진 처리를 위한 폴더들(raw, inter, proc) 생성
            input_dir, intermediate_dir, output_dir = config.setup_photo_paths(photo_specific_name)
            
            # 원본 이미지를 개별 사진 폴더의 'raw' 폴더로 복사
            input_path = os.path.join(input_dir, filename)
            shutil.copy(original_input_path, input_path)
            print(f"'{original_input_path}' -> '{input_path}' 로 복사했습니다.")

        else:
            print("[오류] 잘못된 입력입니다.")
            continue

        # --- 1단계: 이미지 로드 및 확인 ---
        print(f"\n[1단계] 이미지 로드: {input_path}")
        try:
            img_array = np.fromfile(input_path, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[오류] 파일을 읽는 중 에러 발생: {e}")
            continue

        if image is None:
            print(f"[오류] 이미지를 불러올 수 없습니다.")
            continue

        # --- 2단계: 사람 및 얼굴 감지/자르기 ---
        person_image = detect_and_crop_person(image)
        if person_image is None:
            print("[오류] 사람을 감지하지 못했습니다.")
            continue

        # --- 중간 과정 저장 ---
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
        nc_path = os.path.join(output_dir, f"{output_base_name}.nc")
        svg_path = os.path.join(output_dir, f"{output_base_name}.svg")
        
        generate_files_canny(img_blur, nc_path, svg_path)
        
        print(f"\n모든 작업 완료! '{output_base_name}' 이름으로 파일이 저장되었습니다.")
        print("-" * 30)

        # 다음 사진을 위해 카운터 증가
        photo_counter += 1

if __name__ == "__main__":
    main()