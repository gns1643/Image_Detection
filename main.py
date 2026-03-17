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
    generate_gemini_sketch
)

class SketchApp:
    def __init__(self):
        print(">> 심플 G-코드 변환기 (AI 스케치 통합 버전) 시작")
        os.makedirs(config.GENERAL_INPUT_DIR, exist_ok=True)
        config.initialize_session()
        self.photo_counter = 1
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_prompt = None # None이면 기본 프롬프트 사용

    def check_gemini_config(self):
        """Gemini 사용 전 API 키와 프롬프트를 확인 및 설정합니다."""
        if not self.gemini_api_key:
            print("\n[설정] Gemini API 키가 환경변수에 없습니다.")
            self.gemini_api_key = input("API 키를 입력해주세요 (Enter로 건너뛰기 가능하나 기능 제한됨): ").strip()
            if not self.gemini_api_key:
                print("[경고] API 키가 없어 Gemini 기능을 사용할 수 없습니다.")
                return False
        
        print("\n[Gemini 프롬프트 설정]")
        print("1. 기본 프롬프트 (세선화 최적화 정교한 선화)")
        print("2. 사용자 지정 프롬프트 입력")
        p_choice = input("선택 (기본값 1): ").strip()
        
        if p_choice == '2':
            self.gemini_prompt = input("프롬프트를 입력하세요: ").strip()
        else:
            self.gemini_prompt = None # 기본 프롬프트 사용
            
        return True

    def run(self):
        while True:
            choice = self.show_main_menu()
            if choice == 'Q':
                print("프로그램을 종료합니다.")
                break
            
            mode_map = {
                '1': ('AI_ANIME', 'WEBCAM'),
                '2': ('AI_ANIME', 'FILE'),
                '3': ('GEMINI', 'WEBCAM'),
                '4': ('GEMINI', 'FILE')
            }
            
            if choice not in mode_map:
                print("[알림] 잘못된 선택입니다.")
                continue
                
            sketch_type, input_type = mode_map[choice]
            
            # Gemini 모드일 경우 추가 설정 확인
            if sketch_type == 'GEMINI':
                if not self.check_gemini_config():
                    continue
            
            # 2. 이미지 입력 받기
            input_data = self.get_input(input_type)
            if input_data is None:
                continue
            
            input_path, base_filename = input_data
            
            # 3. 이미지 로드 및 크롭 (전처리)
            processed_image, intermediate_dir, output_dir = self.preprocess(input_path, base_filename)
            if processed_image is None:
                continue

            # 4. AI 스케치 생성 및 G-코드 변환
            self.process_and_save(processed_image, sketch_type, base_filename, intermediate_dir, output_dir)
            
            self.photo_counter += 1
            print("-" * 30)

    def show_main_menu(self):
        print("\n" + "="*50)
        print(" [메인 메뉴] 작업 방식을 선택하세요")
        print("="*50)
        print(" 1. Informative-Drawing AI (Anime) + 웹캠 촬영")
        print(" 2. Informative-Drawing AI (Anime) + 파일 불러오기")
        print(" 3. Gemini AI 고품질 스케치 + 웹캠 촬영")
        print(" 4. Gemini AI 고품질 스케치 + 파일 불러오기")
        print(" Q. 프로그램 종료")
        print("="*50)
        return input("선택 (1~4 또는 Q): ").strip().upper()

    def get_input(self, input_type):
        photo_specific_name = f"{self.photo_counter}_capture"
        input_dir, _, _ = config.setup_photo_paths(photo_specific_name)

        if input_type == 'WEBCAM':
            print("\n>> 포토부스 실행 (스페이스바: 촬영, ESC: 취소)")
            captured_path = run_photo_booth(save_dir=input_dir)
            if captured_path is None:
                return None
            return captured_path, "webcam"

        elif input_type == 'FILE':
            files = [f for f in os.listdir(config.GENERAL_INPUT_DIR) if os.path.isfile(os.path.join(config.GENERAL_INPUT_DIR, f))]
            if not files:
                print(f"[알림] '{config.GENERAL_INPUT_DIR}' 폴더에 이미지가 없습니다.")
                return None
            
            print(f"\n목록: {', '.join(files)}")
            filename = input("파일명 입력: ").strip().strip('"')
            src_path = os.path.join(config.GENERAL_INPUT_DIR, filename)
            
            if not os.path.exists(src_path):
                print(f"[오류] 파일이 없습니다: {src_path}")
                return None
            
            dest_path = os.path.join(input_dir, filename)
            shutil.copy(src_path, dest_path)
            return dest_path, os.path.splitext(filename)[0]

    def preprocess(self, input_path, base_filename):
        photo_specific_name = f"{self.photo_counter}_capture"
        _, intermediate_dir, output_dir = config.setup_photo_paths(photo_specific_name)

        img_array = np.fromfile(input_path, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            print("[오류] 이미지를 로드할 수 없습니다.")
            return None, None, None

        person_roi = detect_person_and_get_roi(image)
        cropped = image[person_roi[1]:person_roi[1]+person_roi[3], person_roi[0]:person_roi[0]+person_roi[2]] if person_roi else image
        
        face_roi = detect_face_and_get_roi(cropped)
        if face_roi:
            cropped = cropped[face_roi[1]:face_roi[3], face_roi[0]:face_roi[2]]

        preprocessed = image_processor(cropped)
        if preprocessed is None:
            preprocessed = cropped

        cv2.imencode(".jpg", preprocessed)[1].tofile(os.path.join(intermediate_dir, "preprocessed.jpg"))
        return preprocessed, intermediate_dir, output_dir

    def process_and_save(self, image, sketch_type, base_filename, intermediate_dir, output_dir):
        print(f"\n>> [{sketch_type}] 스케치 생성 시작...")
        
        if sketch_type == 'AI_ANIME':
            sketch = generate_sketch(image)
            threshold_val = 220
        else: # GEMINI
            sketch = generate_gemini_sketch(image, api_key=self.gemini_api_key, prompt=self.gemini_prompt)
            threshold_val = 240

        if sketch is None:
            print("[오류] 스케치 생성에 실패했습니다.")
            return

        suffix = "anime" if sketch_type == 'AI_ANIME' else "gemini"
        output_base = f"{base_filename}_{suffix}"
        cv2.imencode(".png", sketch)[1].tofile(os.path.join(intermediate_dir, f"sketch_{suffix}.png"))

        if len(sketch.shape) == 3:
            sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(sketch, threshold_val, 255, cv2.THRESH_BINARY)
        
        print(f">> [세선화 및 G-코드 생성] {output_base}")
        nc_path = os.path.join(output_dir, f"{output_base}.nc")
        svg_path = os.path.join(output_dir, f"{output_base}.svg")
        
        generate_files_thinning(binary, nc_path, svg_path)
        print(f">> 완료: {output_base}.nc")

if __name__ == "__main__":
    app = SketchApp()
    app.run()
