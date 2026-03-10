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
    generate_sketch,
    generate_ap_sketch,
    create_pipeline_diagram,
    generate_nanobanana_sketch
)

def main():
    print(">> 심플 G-코드 변환기 (MediaPipe + AI 스케치 기반) 시작")

    os.makedirs(config.GENERAL_INPUT_DIR, exist_ok=True)
    config.initialize_session()
    photo_counter = 1
    gemini_api_key = os.environ.get("GEMINI_API_KEY")

    while True:
        print("\n" + "="*50)
        print(" [메인 메뉴] 작업 방식을 선택하세요")
        print("="*50)
        print("1. 기존 Canny 방식 (외곽선 강조)")
        print("2. 기존 세선화 방식 (스케치/일러스트용)")
        print("3. Informative-Drawing 기반 AI 스케치 (추천)")
        print("4. APDrawingGAN2 기반 예술 선화 (NEW/고해상도)")
        print("5. Gemini Nanobanana (Google AI 스케치)")
        print("Q. 프로그램 종료")
        print("="*50)
        
        main_choice = input("선택 (1~5 또는 Q): ").strip().upper()

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

        elif main_choice == '4':
            print("\n[APDrawingGAN2 AI 스케치 하위 메뉴]")
            print("1. 웹캠 촬영 (AP 스케치 + Canny)")
            print("2. 기존 이미지 불러오기 (AP 스케치 + Canny)")
            print("3. 웹캠 촬영 (AP 스케치 + 세선화)")
            print("4. 기존 이미지 불러오기 (AP 스케치 + 세선화)")
            sub_choice = input("선택 (1~4): ").strip()
            mapping = {'1': '9', '2': '10', '3': '11', '4': '12'}
            mode = mapping.get(sub_choice)

        elif main_choice == '5':
            print("\n[Gemini Nanobanana AI 스케치 하위 메뉴]")
            print("1. 웹캠 촬영 (Gemini 스케치 + Canny)")
            print("2. 기존 이미지 불러오기 (Gemini 스케치 + Canny)")
            print("3. 웹캠 촬영 (Gemini 스케치 + 세선화)")
            print("4. 기존 이미지 불러오기 (Gemini 스케치 + 세선화)")
            sub_choice = input("선택 (1~4): ").strip()
            mapping = {'1': '13', '2': '14', '3': '15', '4': '16'}
            mode = mapping.get(sub_choice)

        if mode is None:
            print("[알림] 잘못된 선택이거나 상위 메뉴로 돌아갑니다.")
            continue

        base_photo_name = None
        photo_specific_name = None

        # 웹캠을 사용하는 모드 (1, 5, 7, 9, 11, 13, 15)
        if mode in ['1', '5', '7', '9', '11', '13', '15']:
            base_photo_name = "webcam"
            photo_specific_name = f"{photo_counter}_{base_photo_name}"
            input_dir, intermediate_dir, output_dir = config.setup_photo_paths(photo_specific_name)
            
            print("\n>> 포토부스를 실행합니다. (스페이스바: 촬영, ESC: 취소)")
            captured_path = run_photo_booth(save_dir=input_dir)
            if captured_path is None: continue
            
            input_path = captured_path
            base_filename = os.path.splitext(os.path.basename(input_path))[0]

        # 기존 이미지를 사용하는 모드 (2, 3, 4, 6, 8, 10, 12, 14, 16)
        elif mode in ['2', '3', '4', '6', '8', '10', '12', '14', '16']:
            print(f"\n'{config.GENERAL_INPUT_DIR}' 폴더의 이미지 목록:")
            try:
                files = [f for f in os.listdir(config.GENERAL_INPUT_DIR) if os.path.isfile(os.path.join(config.GENERAL_INPUT_DIR, f))]
                if not files:
                    print("- 이미지가 없습니다. 해당 폴더에 이미지를 추가해주세요.")
                    continue
                for f in files: print(f"- {f}")
            except: continue

            filename = input("파일명을 입력하세요: ").strip().strip('"')
            original_input_path = os.path.join(config.GENERAL_INPUT_DIR, filename)
            if not os.path.exists(original_input_path): continue
            
            base_filename = os.path.splitext(filename)[0]
            photo_specific_name = f"{photo_counter}_{base_filename}"
            input_dir, intermediate_dir, output_dir = config.setup_photo_paths(photo_specific_name)
            
            input_path = os.path.join(input_dir, filename)
            shutil.copy(original_input_path, input_path)
            print(f"'{original_input_path}' -> '{input_path}' 로 복사했습니다.")

        # --- 1단계: 이미지 로드 ---
        print(f"\n[1단계] 이미지 로드: {input_path}")
        try:
            img_array = np.fromfile(input_path, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR if mode != '4' else cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            print(f"[오류] {e}"); continue
        if image is None: continue

        # [모드 4] 단순 세선화
        if mode == '4':
            output_base_name = f"{base_filename}_thinned"
            generate_files_thinning(image, os.path.join(output_dir, f"{output_base_name}.nc"), os.path.join(output_dir, f"{output_base_name}.svg"))
            photo_counter += 1; continue

        # [모드 3] 단순 Canny
        if mode == '3':
            img_blur = image_processor(image)
            output_base_name = f"{base_filename}_canny_full"
            generate_files_canny(img_blur, os.path.join(output_dir, f"{output_base_name}.nc"), os.path.join(output_dir, f"{output_base_name}.svg"))
            photo_counter += 1; continue

        # [공통] 인물 감지 및 크롭
        print("\n>> [2단계] MediaPipe 인물 감지")
        person_roi = detect_person_and_get_roi(image)
        person_image = image[person_roi[1]:person_roi[1]+person_roi[3], person_roi[0]:person_roi[0]+person_roi[2]] if person_roi else image
        
        # 인물 크롭 저장
        if person_roi:
            person_crop_path = os.path.join(intermediate_dir, f"{base_filename}_person_crop.png")
            cv2.imencode(".png", person_image)[1].tofile(person_crop_path)
            print(f"  - 인물 크롭 이미지를 저장했습니다: {person_crop_path}")

        face_roi = detect_face_and_get_roi(person_image)
        face_image = person_image[face_roi[1]:face_roi[3], face_roi[0]:face_roi[2]] if face_roi else person_image

        # 얼굴 크롭 저장
        if face_roi:
            face_crop_path = os.path.join(intermediate_dir, f"{base_filename}_face_crop.png")
            cv2.imencode(".png", face_image)[1].tofile(face_crop_path)
            print(f"  - 얼굴 크롭 이미지를 저장했습니다: {face_crop_path}")

        # --- 전처리 ---
        img_blur = image_processor(face_image)
        if img_blur is None: continue
        
        # --- 변환 로직 (Informative vs APDrawing vs Gemini) ---
        sketch_image = None
        if mode in ['5', '6', '7', '8']:
            sketch_image = generate_sketch(img_blur)
        elif mode in ['9', '10', '11', '12']:
            sketch_image = generate_ap_sketch(img_blur)
        elif mode in ['13', '14', '15', '16']:
            if not gemini_api_key:
                gemini_api_key = input("Gemini API 키를 입력하세요: ").strip()
            if gemini_api_key:
                sketch_image = generate_nanobanana_sketch(img_blur, gemini_api_key)
            else:
                print("[오류] API 키가 없어 작업을 중단합니다.")
                continue

        if sketch_image is not None:
            sketch_path = os.path.join(intermediate_dir, f"{base_filename}_sketch.png")
            cv2.imencode(".png", sketch_image)[1].tofile(sketch_path)
            
            # 이진화
            gray_sketch = cv2.cvtColor(sketch_image, cv2.COLOR_BGR2GRAY) if len(sketch_image.shape) == 3 else sketch_image
            _, binary_sketch = cv2.threshold(gray_sketch, 220, 255, cv2.THRESH_BINARY)
            
            # 이진화 스케치 저장
            binary_sketch_path = os.path.join(intermediate_dir, f"{base_filename}_binary_sketch.png")
            cv2.imencode(".png", binary_sketch)[1].tofile(binary_sketch_path)
            print(f"  - 이진화 스케치 이미지를 저장했습니다: {binary_sketch_path}")
            
            # 후속 처리
            if mode in ['5', '6', '9', '10', '13', '14']: # Canny 방식 (Binary 기반)
                output_name = f"{base_filename}_binary"
                generate_files_binary(binary_sketch, os.path.join(output_dir, f"{output_name}.nc"), os.path.join(output_dir, f"{output_name}.svg"))
            else: # 세선화 방식
                output_name = f"{base_filename}_thinning"
                generate_files_thinning(binary_sketch, os.path.join(output_dir, f"{output_name}.nc"), os.path.join(output_dir, f"{output_name}.svg"))
        
        elif mode in ['1', '2']: # 기본 Canny
            generate_files_canny(img_blur, os.path.join(output_dir, f"{base_filename}_canny.nc"), os.path.join(output_dir, f"{base_filename}_canny.svg"))

        # --- 파이프라인 시각화 생성 ---
        try:
            # 존재할 수 있는 경로들 정의
            person_crop_path = os.path.join(intermediate_dir, f"{base_filename}_person_crop.png")
            face_crop_path = os.path.join(intermediate_dir, f"{base_filename}_face_crop.png")
            sketch_path = os.path.join(intermediate_dir, f"{base_filename}_sketch.png")
            binary_sketch_path = os.path.join(intermediate_dir, f"{base_filename}_binary_sketch.png")
            
            paths = [input_path]
            labels = ["Original"]
            
            if os.path.exists(person_crop_path):
                paths.append(person_crop_path); labels.append("Person Crop")
            if os.path.exists(face_crop_path):
                paths.append(face_crop_path); labels.append("Face Crop")
            if os.path.exists(sketch_path):
                paths.append(sketch_path); labels.append("AI Sketch")
            if os.path.exists(binary_sketch_path):
                paths.append(binary_sketch_path); labels.append("Binary Sketch")
                
            diagram_path = os.path.join(intermediate_dir, f"{base_filename}_pipeline.png")
            create_pipeline_diagram(paths, labels, diagram_path)
        except Exception as e:
            print(f"[시각화 오류] {e}")

        print(f"\n>> 작업 완료! 세션 번호: {photo_counter}")
        print("-" * 30)
        photo_counter += 1

if __name__ == "__main__":
    main()
