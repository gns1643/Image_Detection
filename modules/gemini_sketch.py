import os
import cv2
import numpy as np
from google import genai
from google.genai import types

# --- Gemini API 설정 ---
MODEL_ID = "gemini-3.1-flash-image-preview"

def generate_gemini_sketch(image_bgr: np.ndarray, api_key: str = None, prompt: str = None) -> np.ndarray:
    """
    OpenCV 이미지를 입력받아 Gemini API를 사용하여 세선화에 최적화된 고품질 선화를 생성합니다.
    """
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("\n[오류] GEMINI_API_KEY가 설정되지 않았습니다.")
        return None

    if not prompt:
        prompt = (
            "A high-quality, pure black line art caricature based on the provided image. "
            "The entire drawing is rendered exclusively with lines of exactly the same thickness "
            "(uniform line weight, minimal width) using only solid black ink. "
            "The lines are precise, unwavering, and appear machine-drawn for direct path tracing. "
            "Only solid black lines on a clean white background. "
            "No other colors, gradients, shading, or textures are present. "
            "Minimalist geometric details. Focus purely on the continuity of the lines and the main simplified shape."
        )

    print(f">> [Gemini 스케치] Gemini API({MODEL_ID})를 사용하여 선화 추출을 시작합니다.")

    try:
        # 1. OpenCV 이미지(BGR)를 PNG 바이트로 인코딩
        success, encoded_image = cv2.imencode(".png", image_bgr)
        if not success:
            print("[오류] 이미지 인코딩에 실패했습니다.")
            return None
        image_bytes = encoded_image.tobytes()

        # 2. Gemini 클라이언트 설정
        client = genai.Client(api_key=api_key)

        # 3. Gemini API 호출
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=prompt),
                        types.Part(
                            inline_data=types.Blob(
                                data=image_bytes,
                                mime_type="image/png"
                            )
                        )
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"]
            )
        )

        # 4. 결과 처리 (첫 번째 이미지 추출)
        result_sketch = None
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                nparr = np.frombuffer(part.inline_data.data, np.uint8)
                result_sketch = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                break
            elif part.text:
                print(f"   [Gemini 메시지] {part.text}")

        if result_sketch is None:
            print("   [오류] 이미지가 생성되지 않았습니다.")
            return None

        # 5. 후처리 (원본 크기로 맞추기 및 그레이스케일 변환)
        h, w = image_bgr.shape[:2]
        final_sketch = cv2.resize(result_sketch, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        if len(final_sketch.shape) == 3:
            final_sketch = cv2.cvtColor(final_sketch, cv2.COLOR_BGR2GRAY)

        return final_sketch

    except Exception as e:
        print(f"   [오류 발생] {e}")
        return None
