import sys
import os
import cv2
import numpy as np
import shutil
from modules import (
    config,
    image_processor,
    generate_files_canny,
    generate_files_binary,
    generate_files_thinning,
    detect_person_and_get_roi,
    detect_face_and_get_roi,
    run_photo_booth,
    generate_sketch
)

def main():
    print(">> 심플 G-코드 변환기 (MediaPipe + AI 스케치 기반) 시작")

    os.makedirs(config.GENERAL_INPUT_DIR, exist_ok=True)
    config.initialize_session()
    photo_counter = 1

    while True:
        print("\n" + "="*50)
        print(" [메인 메뉴] 작업 방식을 선택하세요")
        print("="*50)
        print("1. 기존 Canny 방식 (외곽선 강조)")
        print("2. 기존 세선화 방식 (스케치/일러스트용)")
        print("3. Informative-Drawing 기반 AI 스케치 (추천)")
        print("Q. 프로그램 종료")
        print("="*50)
        
        main_choice = input("선택 (1~3 또는 Q): ").strip().upper()

        if main_choice == 'Q':
            print("프로그램을 종료합니다.")
            break
        
        mode = None
        if main_choice == '1':
            print("\n[Canny 방식 하위 메뉴]")
            print("1. 웹캠 촬영 (인물 크롭)")
            print("2. 기존 이미지 불러오기 (인물 크롭)")
            print("3. 기존 이미지 불러오기 (크롭 없음, 전체 외곽선)")
            sub_choice = input("선택 (1~3): ").strip()
            if sub_choice in ['1', '2', '3']: mode = sub_choice
            
        elif main_choice == '2':
            print("\n[세선화 방식 하위 메뉴]")
            print("1. 기존 이미지 불러오기 (스케치/일러스트 세선화)")
            sub_choice = input("선택 (1): ").strip()
            if sub_choice == '1': mode = '4'
            
        elif main_choice == '3':
            print("\n[Informative-Drawing AI 스케치 하위 메뉴]")
            print("1. 웹캠 촬영 (AI 스케치 + Canny)")
            print("2. 기존 이미지 불러오기 (AI 스케치 + Canny)")
            print("3. 웹캠 촬영 (AI 스케치 + 세선화)")
            print("4. 기존 이미지 불러오기 (AI 스케치 + 세선화)")
            sub_choice = input("선택 (1~4): ").strip()
            mapping = {'1': '5', '2': '6', '3': '7', '4': '8'}
            mode = mapping.get(sub_choice)

        if mode is None:
            print("[알림] 잘못된 선택이거나 상위 메뉴로 돌아갑니다.")
            continue

        base_photo_name = None
        photo_specific_name = None

        # 웹캠을 사용하는 모드 (1, 5, 7)
        if mode in ['1', '5', '7']:
            base_photo_name = "webcam"
            photo_specific_name = f"{photo_counter}_{base_photo_name}"
            
            input_dir, intermediate_dir, output_dir = config.setup_photo_paths(photo_specific_name)
            
            print("\n>> 포토부스를 실행합니다. (스페이스바: 촬영, ESC: 취소)")
            captured_path = run_photo_booth(save_dir=input_dir)
            
            if captured_path is None:
                print("[알림] 촬영이 취소되었습니다.")
                continue
            
            input_path = captured_path
            base_filename = os.path.splitext(os.path.basename(input_path))[0]

        # 기존 이미지를 사용하는 모드 (2, 3, 4, 6, 8)
        elif mode in ['2', '3', '4', '6', '8']:
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

            input_dir, intermediate_dir, output_dir = config.setup_photo_paths(photo_specific_name)
            
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
            if mode == '4':
                image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[오류] 파일을 읽는 중 에러 발생: {e}")
            continue

        if image is None:
            print(f"[오류] 이미지를 불러올 수 없습니다.")
            continue

        # [모드 4] 스케치/일러스트 처리 (세선화)
        if mode == '4':
            output_base_name = f"{base_filename}_thinned"
            nc_path = os.path.join(output_dir, f"{output_base_name}.nc")
            svg_path = os.path.join(output_dir, f"{output_base_name}.svg")
            
            generate_files_thinning(image, nc_path, svg_path)
            
            print(f"\n>> 모든 작업 완료! '{output_base_name}' 이름으로 파일이 저장되었습니다.")
            print("-" * 30)
            photo_counter += 1
            continue

        # [모드 3] 크롭 없이 Canny 외곽선 따기
        if mode == '3':
            img_blur = image_processor(image)
            if img_blur is None:
                print("[오류] 이미지 전처리 중 문제가 발생했습니다.")
                continue
            
            output_base_name = f"{base_filename}_canny_full"
            nc_path = os.path.join(output_dir, f"{output_base_name}.nc")
            svg_path = os.path.join(output_dir, f"{output_base_name}.svg")
            
            generate_files_canny(img_blur, nc_path, svg_path)
            
            print(f"\n>> 모든 작업 완료! '{output_base_name}' 이름으로 파일이 저장되었습니다.")
            print("-" * 30)
            photo_counter += 1
            continue

        # [모드 1, 2, 5, 6, 7, 8] 인물 사진 처리 (자동 크롭)
        print("\n>> [2단계] MediaPipe로 상체 및 얼굴 감지 시작")
        person_roi = detect_person_and_get_roi(image)
        
        if person_roi:
            print("   - 상체 감지 완료. 이미지를 자릅니다.")
            x, y, w, h = person_roi
            person_image = image[y:y+h, x:x+w]
        else:
            print("   - 상체를 감지하지 못했습니다. 전체 이미지를 사용합니다.")
            person_image = image

        person_intermediate_filename = f"{base_filename}_cropped_person.jpg"
        person_intermediate_path = os.path.join(intermediate_dir, person_intermediate_filename)
        is_success_person, im_buf_arr_person = cv2.imencode(".jpg", person_image)
        if is_success_person:
            im_buf_arr_person.tofile(person_intermediate_path)
            print(f"   - 중간 저장(1. 상체 크롭): '{person_intermediate_path}'")

        face_roi = detect_face_and_get_roi(person_image)
        if face_roi:
            print("   - 얼굴 감지 완료. A5 비율로 이미지를 다시 자릅니다.")
            x1, y1, x2, y2 = face_roi
            face_image = person_image[y1:y2, x1:x2]
        else:
            print("   - 얼굴을 감지하지 못했습니다. 상체 이미지를 그대로 사용합니다.")
            face_image = person_image

        face_intermediate_filename = f"{base_filename}_cropped_face.jpg"
        face_intermediate_path = os.path.join(intermediate_dir, face_intermediate_filename)
        is_success_face, im_buf_arr_face = cv2.imencode(".jpg", face_image)
        if is_success_face:
            im_buf_arr_face.tofile(face_intermediate_path)
            print(f"   - 중간 저장(2. 얼굴 크롭): '{face_intermediate_path}'")

        # --- 3단계: 이미지 전처리 (배경 제거 + 블러) ---
        print("\n>> [3단계] 이미지 전처리 시작")
        img_blur = image_processor(face_image)
        if img_blur is None:
            print("[오류] 이미지 전처리 중 문제가 발생했습니다.")
            continue
            
        bg_removed_path = os.path.join(intermediate_dir, f"{base_filename}_3_bg_removed.png")
        is_success_bg, im_buf_bg = cv2.imencode(".png", img_blur)
        if is_success_bg:
            im_buf_bg.tofile(bg_removed_path)
            print(f"   - 중간 저장(3. 배경 제거): '{bg_removed_path}'")
        
        # --- 4단계: 변환 및 저장 ---
        if mode in ['1', '2']:
            output_base_name = f"{base_filename}_canny_face"
            nc_path = os.path.join(output_dir, f"{output_base_name}.nc")
            svg_path = os.path.join(output_dir, f"{output_base_name}.svg")
            
            generate_files_canny(img_blur, nc_path, svg_path)
            print(f"\n>> 모든 작업 완료! '{output_base_name}' 이름으로 파일이 저장되었습니다.")

        elif mode in ['5', '6']:
            print("\n>> [3.5단계] 딥러닝 AI 스케치 변환 (Anime 스타일) 시작")
            sketch_image = generate_sketch(img_blur)
            if sketch_image is None: continue
                
            sketch_path = os.path.join(intermediate_dir, f"{base_filename}_4_ai_sketch_anime.png")
            is_success_sketch, im_buf_sketch = cv2.imencode(".png", sketch_image)
            if is_success_sketch:
                im_buf_sketch.tofile(sketch_path)
                print(f"   - 중간 저장(4. AI 스케치): '{sketch_path}'")
            
            # --- 추가된 로직: 이진화 처리 및 중간 저장 ---
            print("\n>> [4단계] 이진화(Binarization) 및 중간 저장 시작")
            
            if len(sketch_image.shape) == 3:
                gray_sketch = cv2.cvtColor(sketch_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_sketch = sketch_image
                
            # 임계값 200: 숫자를 낮추면 진한 선만, 높이면 연한 선도 포함됩니다.
            _, binary_sketch = cv2.threshold(gray_sketch, 220, 255, cv2.THRESH_BINARY)
            
            binary_path = os.path.join(intermediate_dir, f"{base_filename}_5_binary.png")
            is_success_bin, im_buf_bin = cv2.imencode(".png", binary_sketch)
            if is_success_bin:
                im_buf_bin.tofile(binary_path)
                print(f"   - 중간 저장(5. 이진화): '{binary_path}'")
            # ----------------------------------------------

            print("\n>> [5단계] 외곽선 추출 및 G-코드 생성 시작")
            output_base_name = f"{base_filename}_ai_sketch_anime_binary"
            nc_path = os.path.join(output_dir, f"{output_base_name}.nc")
            svg_path = os.path.join(output_dir, f"{output_base_name}.svg")
            
            # Canny 대신 이진화 이미지 전용 함수로 G코드 생성
            generate_files_binary(binary_sketch, nc_path, svg_path)
            print(f"\n>> 모든 작업 완료! '{output_base_name}' 이름으로 파일이 저장되었습니다.")

        elif mode in ['7', '8']:
            print("\n>> [3.5단계] 딥러닝 AI 스케치 변환 (Anime 스타일) 시작")
            sketch_image = generate_sketch(img_blur)
            if sketch_image is None: continue
                
            sketch_path = os.path.join(intermediate_dir, f"{base_filename}_4_ai_sketch_anime.png")
            is_success_sketch, im_buf_sketch = cv2.imencode(".png", sketch_image)
            if is_success_sketch:
                im_buf_sketch.tofile(sketch_path)
                print(f"   - 중간 저장(4. AI 스케치): '{sketch_path}'")
            
            # --- [수정된 부분] 1회만 이진화 수행 및 저장 ---
            print("\n>> [4단계] 이진화(Binarization) 및 중간 저장 시작")
            if len(sketch_image.shape) == 3:
                gray_sketch = cv2.cvtColor(sketch_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_sketch = sketch_image
                
            # 눈으로 확인하기 편하도록 흰 바탕에 검은 선(THRESH_BINARY)으로 만듭니다.
            _, binary_sketch = cv2.threshold(gray_sketch, 220, 255, cv2.THRESH_BINARY)
            
            binary_path = os.path.join(intermediate_dir, f"{base_filename}_5_binary.png")
            is_success_bin, im_buf_bin = cv2.imencode(".png", binary_sketch)
            if is_success_bin:
                im_buf_bin.tofile(binary_path)
                print(f"   - 중간 저장(5. 이진화): '{binary_path}'")
            # ---------------------------------------------------
            
            output_base_name = f"{base_filename}_ai_sketch_anime_thinning"
            nc_path = os.path.join(output_dir, f"{output_base_name}.nc")
            svg_path = os.path.join(output_dir, f"{output_base_name}.svg")
            
            print("\n>> [5단계] 세선화(Thinning) 및 G-코드 생성 시작")
            # 원본 스케치 대신 '이진화가 완료된 이미지'를 모듈로 넘깁니다.
            generate_files_thinning(binary_sketch, nc_path, svg_path)
            print(f"\n>> 모든 작업 완료! '{output_base_name}' 이름으로 파일이 저장되었습니다.")

        print("-" * 30)
        photo_counter += 1

if __name__ == "__main__":
    main()