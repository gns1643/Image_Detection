import os
from google import genai
from google.genai import types

def main():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("\n에러: GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("-" * 50)
        print("[터미널 설정 방법]")
        print("1. Command Prompt (CMD):")
        print("   set GEMINI_API_KEY=your_api_key_here")
        print("\n2. PowerShell:")
        print('   $env:GEMINI_API_KEY = "your_api_key_here"')
        print("-" * 50)
        return

    client = genai.Client(api_key=api_key)

    # ✅ 실제 존재하는 이미지 생성/편집 지원 모델
    model_id = "gemini-3.1-flash-image-preview"

    print(f"\n[이미지 수정기 - 모델: {model_id}]")

    image_path = input("원본 이미지 파일 경로를 입력하세요: ").strip()
    if not os.path.exists(image_path):
        print(f"에러: 파일 '{image_path}'를 찾을 수 없습니다.")
        return

    # 고정 프롬프트
    prompt = (
        "A high-quality, pure black line art caricature based on the provided image. "
        "The entire drawing is rendered exclusively with lines of exactly the same thickness "
        "(uniform line weight, minimal width) using only solid black ink. "
        "The lines are precise, unwavering, and appear machine-drawn for direct path tracing. "
        "Only solid black lines on a clean white background. "
        "No other colors, gradients, shading, or textures are present. "
        "Minimalist geometric details. Focus purely on the continuity of the lines and the main simplified shape."
    )

    print("\n지정된 스타일(Pure Black Line Art)로 이미지 수정을 시작합니다...")

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # mime_type 자동 감지
        ext = image_path.lower().split(".")[-1]
        mime_map = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp"
        }
        mime_type = mime_map.get(ext, "image/png")

        # ✅ interactions 대신 generate_content 사용
        response = client.models.generate_content(
            model=model_id,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=prompt),
                        types.Part(
                            inline_data=types.Blob(
                                data=image_bytes,
                                mime_type=mime_type
                            )
                        )
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"]
            )
        )

        image_count = 0
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                file_name = f"caricature_result_{image_count}.png"
                with open(file_name, "wb") as f:
                    f.write(part.inline_data.data)
                print(f"\n✅ 성공! 결과 이미지: '{file_name}'")
                try:
                    os.startfile(file_name)
                except:
                    pass
                image_count += 1
            elif part.text:
                print(f"Gemini 메시지: {part.text}")

        if image_count == 0:
            print("\n이미지가 생성되지 않았습니다.")

    except Exception as e:
        print(f"\n에러 발생: {e}")

if __name__ == "__main__":
    main()