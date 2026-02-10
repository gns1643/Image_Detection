import sys
import os
import cv2
import numpy as np
from modules import (
    config, 
    preprocess_image, 
    generate_files_canny, 
    detect_and_crop_person, 
    detect_and_crop_face
)

def main():
    print(">> 심플 G-코드 변환기 (얼굴 인식 포함) 시작")
    
    # --- 입력 받기 ---
    filename = input(f"'{config.INPUT_DIR}' 폴더에 있는 이미지 파일명을 입력하세요: ").strip().strip('"')
    input_path = os.path.join(config.INPUT_DIR, filename)

    # --- 1단계: 이미지 로드 및 확인 ---
    print(f"[1단계] 이미지 로드: {input_path}")
    if not os.path.exists(input_path):
        print(f"[오류] 오류: 입력 파일이 없습니다: {input_path}")
        return

    img_array = np.fromfile(input_path, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image is None:
        print(f"[오류] 오류: 이미지를 불러올 수 없습니다. 파일이 손상되었거나 이미지 파일이 아닐 수 있습니다.")
        return

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # --- 2단계: 사람 및 얼굴 감지/자르기 ---
    person_image = detect_and_crop_person(image)
    face_image = detect_and_crop_face(person_image)

    # --- 중간 과정 저장: 잘라낸 얼굴 ---
    intermediate_dir = 'data/intermediate'
    os.makedirs(intermediate_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(input_path))[0]
    intermediate_filename = f"{base_filename}_cropped_face.jpg"
    intermediate_path = os.path.join(intermediate_dir, intermediate_filename)

    # 한글 경로 및 파일명 처리를 위해 cv2.imencode 사용
    is_success, im_buf_arr = cv2.imencode(".jpg", face_image)
    if is_success:
        im_buf_arr.tofile(intermediate_path)
        print(f"중간 저장: 잘라낸 얼굴 이미지를 '{intermediate_path}'에 저장했습니다.")

    # --- 3단계: 이미지 전처리 (배경 제거 + 블러) ---
    img_blur = preprocess_image(face_image)
    if img_blur is None: 
        print("[오류] 오류: 이미지 전처리 중 문제가 발생했습니다.")
        return
    
    # --- 4단계: 변환 및 저장 (Canny 방식) ---
    base_filename = os.path.splitext(os.path.basename(input_path))[0]
    # 출력 파일명에 '_face'를 추가하여 구분
    output_base_name = f"{base_filename}_face"
    nc_path = os.path.join(config.OUTPUT_DIR, f"{output_base_name}.nc")
    svg_path = os.path.join(config.OUTPUT_DIR, f"{output_base_name}.svg")
    
    generate_files_canny(img_blur, nc_path, svg_path)
    
    print(f"\n모든 작업 완료! '{output_base_name}' 이름으로 파일이 저장되었습니다.")

if __name__ == "__main__":
    main()