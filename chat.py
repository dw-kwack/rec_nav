import os
from openai import OpenAI
import json

# 환경변수에서 API 키 불러오기
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# 글로벌 리스트 초기화 (누적 저장)
direct = []
indirect = []

def extract_user_interest(user_text: str) -> dict:
    """
    사용자의 텍스트 입력을 받아 관심사를 추출합니다.
    CSV에 있는 작품 이름이 직접 언급된 경우는 반드시 'direct' 항목에 포함하고, 
    간접적 키워드는 'indirect' 항목에 포함시키도록 요청합니다.
    
    반환 예시:
    {
      "direct": ["Mona Lisa"],
      "indirect": ["history of religion", "fossil"]
    }
    """
    prompt = (
        "다음 텍스트에서 사용자가 언급한 관심사를 추출해줘.\n"
        "CSV에 있는 작품 이름이 직접 언급된 경우는 반드시 'direct' 항목에 포함시키고, "
        "간접적 키워드는 'indirect' 항목에 포함시켜줘.\n"
        "예시 입력: \"I am interested in Mona Lisa and I want to see fossil.\"\n"
        "출력은 반드시 아래 JSON 형식이어야 해:\n"
        '{ "direct": ["작품이름1", ...], "indirect": ["키워드1", ...] }'
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "너는 사용자의 관심사를 추출하는 전문가야."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": f"사용자 입력: {user_text}"}
        ],
        temperature=0.7,
        max_tokens=150
    )
    
    result_json = response.choices[0].message.content.strip()
    try:
        interest_data = json.loads(result_json)
    except Exception as e:
        print("JSON 파싱 오류:", e)
        interest_data = {"direct": [], "indirect": []}
    return interest_data

if __name__ == "__main__":
    while True:
        user_input = input("사용자 관심사를 입력하세요 (종료하려면 'exit' 입력): ")
        if user_input.lower() == "exit":
            print("프로그램을 종료합니다.")
            break
        
        interest = extract_user_interest(user_input)
        print("추출된 관심사:")
        print(json.dumps(interest, indent=2, ensure_ascii=False))
        
        # 새로운 관심사가 있으면 글로벌 리스트에 추가 (중복 제거)
        if "direct" in interest:
            for item in interest["direct"]:
                if item not in direct:
                    direct.append(item)
        if "indirect" in interest:
            for item in interest["indirect"]:
                if item not in indirect:
                    indirect.append(item)
        
        print("누적된 direct:", direct)
        print("누적된 indirect:", indirect)
