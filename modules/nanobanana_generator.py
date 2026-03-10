import os
import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
import io
import time
from google.api_core import exceptions

def generate_nanobanana_sketch(image_bgr: np.ndarray, api_key: str) -> np.ndarray:
    """
    Google Gemini 모델을 사용하여 이미지를 고퀄리티 예술적 스케치로 변환합니다.
    쿼터 초과 시 재시도 및 모델 폴백(Fallback) 로직을 포함합니다.
    """
    print(">> [나노바나나 변환] Gemini 모델을 통한 변환을 시작합니다.")
    
    if not api_key:
        print("[오류] Gemini API 키가 설정되지 않았습니다.")
        return None

    # 시도할 모델 리스트 (우선순위 순)
    models_to_try = ['gemini-2.0-flash', 'gemini-1.5-flash']
    max_retries = 2
    retry_delay = 5 # 초

    # OpenCV BGR -> PIL Image (RGB)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    
    # 바이트 스트림으로 변환
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='PNG')
    img_data = img_byte_arr.getvalue()

    # 프롬프트 구성
    prompt = (
        "Transform this person image into a high-quality, professional artistic line drawing sketch. "
        "Use clean, expressive lines. The output should be a monochrome (black lines on white background) "
        "image that is suitable for a pen plotter or CNC drawing machine. "
        "Ensure the facial features and character essence are well-preserved. "
        "Return ONLY the transformed image."
    )

    genai.configure(api_key=api_key)

    for model_name in models_to_try:
        print(f"  - [{model_name}] 모델 시도 중...")
        
        for attempt in range(max_retries + 1):
            try:
                model = genai.GenerativeModel(model_name)
                
                # 이미지 전송 및 생성
                response = model.generate_content([
                    prompt,
                    {'mime_type': 'image/png', 'data': img_data}
                ])

                if not response.parts:
                    print(f"  [경고] {model_name} 모델로부터 응답을 받지 못했습니다.")
                    break # 다음 모델로 넘어감

                image_part = None
                # 후보군(candidates) 확인
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            image_part = part.inline_data
                            break
                
                if image_part:
                    nparr = np.frombuffer(image_part.data, np.uint8)
                    output_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    print(f"  - [{model_name}] 변환 성공!")
                    return output_img
                else:
                    print(f"  [알림] {model_name}가 이미지를 직접 반환하지 않았습니다. 텍스트 응답 확인:")
                    # 텍스트 응답이 있으면 출력 (디버깅용)
                    try:
                        print(response.text[:200] + "...") 
                    except:
                        pass
                    break # 다음 모델 시도

            except exceptions.ResourceExhausted as e:
                print(f"  [쿼터 초과] {model_name}: {e}")
                if attempt < max_retries:
                    print(f"  {retry_delay}초 후 다시 시도합니다... ({attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    print(f"  {model_name} 시도 횟수 초과.")
                    break # 다음 모델로 넘어감

            except exceptions.InvalidArgument as e:
                print(f"  [인자 오류] API 키가 잘못되었거나 설정이 올바르지 않습니다: {e}")
                return None # 이 오류는 모델을 바꿔도 해결되지 않을 가능성이 높음

            except Exception as e:
                print(f"  [기타 오류] {model_name}: {e}")
                break # 다음 모델 시도

    print("[최종 오류] 모든 사용 가능한 Gemini 모델의 쿼터가 초과되었거나 응답이 없습니다.")
    print("팁: 1분 정도 기다린 후 다시 시도하거나, 다른 API 키를 사용해 보세요.")
    return None
