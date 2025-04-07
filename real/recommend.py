# recommend.py
import os
import json
from openai import OpenAI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from embedding import load_csv_and_generate_embeddings, get_embedding

def load_shared_data(filename="shared_data.json") -> dict:
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {"user_info": [], "direct": [], "indirect": []}

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# 공유 데이터 로드: caption.py의 user_info, chat.py의 direct, indirect
shared_data = load_shared_data()
direct = shared_data.get("direct", [])
indirect = shared_data.get("indirect", [])
user_info = shared_data.get("user_info", [])

# 사용자 관심사 합치기
combined_interests = list(set(direct + indirect + user_info))
query_text = " ".join(combined_interests)
print("사용자 관심사 질의:", query_text)

# 질의 임베딩 생성
query_embedding = get_embedding(query_text)

# CSV 파일에서 임베딩 공간 로드
csv_path = '/home/usr/Desktop/recnav/gpt/item_concat.csv'  # 실제 CSV 파일 경로에 맞게 수정
df = load_csv_and_generate_embeddings(csv_path)
print("CSV 임베딩 공간 로드 완료. 전체 작품 수:", len(df))

# 각 작품 임베딩과 질의 임베딩 간 코사인 유사도 계산
df["similarity"] = df["embedding"].apply(lambda emb: cosine_similarity([emb], [query_embedding])[0][0])
n = 5  # 추천할 작품 개수
df_top = df.sort_values("similarity", ascending=False).head(n)

# 추천 후보 작품 정보를 문자열로 구성 (출력에 사용할 정보: name, coordinate)
candidate_info = ""
for _, row in df_top.iterrows():
    candidate_info += f"작품명: {row['name']}, 좌표: {row['coordinate']}\n"
print("추천 후보 작품 목록:\n", candidate_info)

# 추천 프롬프트 구성
recommend_prompt = (
    "[참조 자료]\n"
    "아래는 추천 후보 작품 목록입니다:\n"
    f"{candidate_info}\n"
    "[요청]\n"
    "위 후보 작품들 중에서 사용자의 관심사를 고려하여 반드시 " + str(n) + "개의 작품을 추천해 주세요.\n"
    "추천 결과는 반드시 아래 JSON 형식에 맞게 출력해 주세요:\n"
    '{"recommendations": [{"name": "작품명", "coordinate": "좌표"}]}\n'
    "출력은 오직 유효한 JSON이어야 합니다."
)

rec_response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "너는 박물관 작품 추천 시스템 전문가야."},
        {"role": "user", "content": recommend_prompt}
    ],
    temperature=0.7,
    max_tokens=300
)

raw_output = rec_response.choices[0].message.content.strip()

def clean_json_output(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text

cleaned_output = clean_json_output(raw_output)

try:
    recommendation_data = json.loads(cleaned_output)
    print("추천 결과:")
    print(json.dumps(recommendation_data, indent=2, ensure_ascii=False))
except json.JSONDecodeError:
    print("추천 결과가 올바른 JSON 형식이 아닙니다:")
    print(cleaned_output)
