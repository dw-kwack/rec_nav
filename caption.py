# captioning.py
import os
from openai import OpenAI
import json

# 환경변수에서 API 키 불러오기
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# 전역 변수로 user_information (캡션 내의 'what' 정보) 저장
user_information = []

def caption_image(image_base64: str) -> dict:
    """
    이미지(base64 인코딩된 문자열)를 받아 OpenAI API를 통해 캡션을 생성하고,
    구조화된 출력을 반환.
    
    예시 구조:
    {
      "captions": [
          {"who": "he", "what": "Mona Lisa"},
          {"who": "she", "what": "impressionism"}
      ]
    }
    """
    prompt = (
        "박물관 추천 프롬프트:\n"
        "입력된 이미지에 대해 구조화된 캡션을 생성해줘.\n"
        "출력은 반드시 아래 JSON 형식이어야 해:\n"
        '{ "captions": [ { "who": "문자열", "what": "문자열" }, ... ] }\n'
        "예시: \"he looking at a Mona Lisa\", \"she looking at a impressionism\""
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "너는 박물관 캡셔닝 전문가야. 이미지에 대해 구조화된 캡션을 생성해줘."},
            {"role": "user", "content": prompt},
            # 이미지 부분: 실제 환경에서는 image 전송 포맷에 맞게 전송
            {"role": "user", "content": f"[이미지 데이터: {image_base64}]"}
        ],
        temperature=0.7,
        max_tokens=200
    )
    
    # 응답에서 구조화된 캡션 JSON 문자열을 받아 dict로 변환
    caption_json = response['choices'][0]['message']['content']
    try:
        caption_data = json.loads(caption_json)
    except Exception as e:
        print("JSON 파싱 오류:", e)
        caption_data = {}
    return caption_data

def update_user_information(caption_data: dict):
    """
    캡션 데이터에서 'what' 정보를 추출하여 전역 변수 user_information에 추가.
    """
    global user_information
    if "captions" in caption_data:
        for cap in caption_data["captions"]:
            # 중복 없이 추가 (원하는 로직에 따라 수정 가능)
            if "what" in cap and cap["what"] not in user_information:
                user_information.append(cap["what"])
    print("현재 user_information:", user_information)

if __name__ == "__main__":
    # 테스트용: 임의의 이미지 base64 문자열 (실제 환경에서는 이미지 파일을 인코딩하여 사용)
    test_image_base64 = "dummy_base64_encoded_image_string"
    
    caption_result = caption_image(test_image_base64)
    print("캡션 결과:", caption_result)
    
    update_user_information(caption_result)
