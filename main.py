import os
import cv2
import numpy as np
import shutil
import json
from concurrent.futures import ThreadPoolExecutor
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
        self.gemini_prompt = None
        
        # 프롬프트 설정 파일 경로
        self.prompts_file = os.path.join(config.BASE_DIR, "config", "gemini_prompts.json")
        self.prompts_dict = self.load_prompts()

    def load_prompts(self):
        """JSON 파일에서 프롬프트 목록을 로드합니다."""
        if not os.path.exists(self.prompts_file):
            print(f"\n[오류] 프롬프트 설정 파일이 없습니다: {self.prompts_file}")
            print("      'config/gemini_prompts.json' 파일을 생성하고 스타일을 추가해주세요.")
            return {}
            
        try:
            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not data:
                    print(f"[경고] 프롬프트 파일이 비어 있습니다: {self.prompts_file}")
                    return {}
                return data
        except Exception as e:
            print(f"[오류] 프롬프트 파일을 읽는 중 오류 발생: {e}")
            return {}

    def check_gemini_config(self):
        """Gemini 사용 전 API 키와 프롬프트를 확인 및 설정합니다."""
        if not self.gemini_api_key:
            print("\n[설정] Gemini API 키가 환경변수에 없습니다.")
            self.gemini_api_key = input("API 키를 입력해주세요 (Enter로 건너뛰기 가능하나 기능 제한됨): ").strip()
            if not self.gemini_api_key:
                print("[경고] API 키가 없어 Gemini 기능을 사용할 수 없습니다.")
                return False
        
        self.prompts_dict = self.load_prompts()
        prompt_keys = list(self.prompts_dict.keys())

        if not prompt_keys:
            print("\n" + "-"*40)
            print(" [알림] 등록된 Gemini 스타일이 없습니다.")
            print(" 1. 사용자 직접 입력")
            print(" Q. 메뉴로 돌아가기")
            print("-"*40)
            
            p_input = input("선택 (1 또는 Q): ").strip().upper()
            if p_input == '1':
                self.gemini_prompt = input("프롬프트를 직접 입력하세요: ").strip()
                return True if self.gemini_prompt else False
            else:
                return False

        print("\n" + "-"*40)
        print(" [Gemini 스타일 선택]")
        print("-"*40)
        print(" 0. 모든 프롬프트 순차 실행 (Batch Mode)")
        for i, key in enumerate(prompt_keys, 1):
            print(f" {i}. {key}")
        print(f" {len(prompt_keys) + 1}. 사용자 직접 입력")
        print("-"*40)
        
        try:
            p_input = input(f"선택 (0~{len(prompt_keys) + 1}, 기본값 1): ").strip()
            if p_input == '0':
                self.gemini_prompt = self.prompts_dict # 딕셔너리 전체 전달
                print(f">> [배치 모드] 모든 {len(prompt_keys)}개 스타일을 순차적으로 하나씩 실행합니다.")
            elif not p_input:
                self.gemini_prompt = self.prompts_dict[prompt_keys[0]]
                print(f">> '1. {prompt_keys[0]}' 스타일(기본값)이 적용되었습니다.")
            else:
                p_choice = int(p_input)
                if 1 <= p_choice <= len(prompt_keys):
                    selected_key = prompt_keys[p_choice - 1]
                    self.gemini_prompt = self.prompts_dict[selected_key]
                    print(f">> '{p_choice}. {selected_key}' 스타일이 적용되었습니다.")
                elif p_choice == len(prompt_keys) + 1:
                    self.gemini_prompt = input("프롬프트를 직접 입력하세요: ").strip()
                else:
                    self.gemini_prompt = self.prompts_dict[prompt_keys[0]]
        except (ValueError, IndexError):
            self.gemini_prompt = self.prompts_dict[prompt_keys[0]]
            
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
            
            if sketch_type == 'GEMINI':
                if not self.check_gemini_config():
                    continue
            
            input_data = self.get_input(input_type)
            if input_data is None:
                continue
            
            input_path, base_filename = input_data
            processed_image, intermediate_dir, output_dir = self.preprocess(input_path, base_filename)
            if processed_image is None:
                continue

            # 4. AI 스케치 생성 및 G-코드 변환 (Batch Mode 대응)
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
        """AI 스케치 생성 후 세선화를 거쳐 G-코드로 저장합니다. (Batch 모드 지원)"""
        
        # Gemini 배치 모드 처리 (딕셔너리인 경우)
        if sketch_type == 'GEMINI' and isinstance(self.gemini_prompt, dict):
            print(f"\n>> [일괄 순차 처리 시작] 총 {len(self.gemini_prompt)}개의 스타일을 하나씩 적용합니다.")
            
            for i, (style_name, prompt_text) in enumerate(self.gemini_prompt.items(), 1):
                # 파일명에 사용할 안전한 스타일 이름 생성
                safe_style_name = "".join([c if c.isalnum() else "_" for c in style_name])
                current_base = f"{base_filename}_{safe_style_name}"
                
                print(f"\n[{style_name}] ({i}/{len(self.gemini_prompt)}) 작업 시작...")
                self._single_process_and_save(image, sketch_type, prompt_text, current_base, intermediate_dir, output_dir, style_name)
        else:
            # 단일 모드 처리
            self._single_process_and_save(image, sketch_type, self.gemini_prompt, base_filename, intermediate_dir, output_dir, "단일 모드")

    def _single_process_and_save(self, image, sketch_type, prompt, base_filename, intermediate_dir, output_dir, style_name="Default"):
        """실제 한 장의 이미지를 변환하고 저장하는 내부 메서드"""
        if sketch_type == 'AI_ANIME':
            sketch = generate_sketch(image)
            threshold_val = 220
            suffix = "anime"
        else: # GEMINI
            sketch = generate_gemini_sketch(image, api_key=self.gemini_api_key, prompt=prompt, style_name=style_name)
            threshold_val = 240
            suffix = "gemini"

        if sketch is None:
            print(f"[{style_name}] [오류] '{base_filename}' 스케치 생성 실패")
            return

        output_base = f"{base_filename}_{suffix}"
        cv2.imencode(".png", sketch)[1].tofile(os.path.join(intermediate_dir, f"{output_base}.png"))

        if len(sketch.shape) == 3:
            sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(sketch, threshold_val, 255, cv2.THRESH_BINARY)
        
        print(f"[{style_name}] >> [세선화 및 G-코드 생성] {output_base}")
        nc_path = os.path.join(output_dir, f"{output_base}.nc")
        svg_path = os.path.join(output_dir, f"{output_base}.svg")
        
        generate_files_thinning(binary, nc_path, svg_path)
        print(f"[{style_name}] >> 완료: {output_base}.nc")

if __name__ == "__main__":
    app = SketchApp()
    app.run()
