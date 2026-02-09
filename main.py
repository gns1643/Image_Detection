import sys
import os

# 현재 폴더 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules import config
from modules import load_and_preprocess, generate_files_canny

def main():
    print("🚀 심플 G-코드 변환기 (Canny Only) 시작")
    
    # 1. 경로 확인
    if not os.path.exists(config.INPUT_DIR):
        os.makedirs(config.INPUT_DIR, exist_ok=True)
        print(f"❌ 입력 폴더를 생성했습니다: {config.INPUT_DIR}")
        print("   -> 여기에 'input.png' 파일을 넣어주세요.")
        return

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    input_path = os.path.join(config.INPUT_DIR, config.INPUT_FILENAME)

    if not os.path.exists(input_path):
        print(f"❌ 오류: 입력 파일이 없습니다: {input_path}")
        return

    # 2. 이미지 전처리 (배경 제거 + 블러)
    img_blur = load_and_preprocess(input_path)
    if img_blur is None: return
    
    # 3. 변환 및 저장 (Canny 방식)
    nc_path = os.path.join(config.OUTPUT_DIR, config.OUTPUT_NC_FILENAME)
    svg_path = os.path.join(config.OUTPUT_DIR, config.OUTPUT_SVG_FILENAME)
    
    generate_files_canny(img_blur, nc_path, svg_path)
    
    print("\n🎉 모든 작업 완료!")

if __name__ == "__main__":
    main()