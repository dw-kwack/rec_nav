from openai import OpenAI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

csv_path = '/home/usr/Desktop/recnav/gpt/item_concat.csv'
df = pd.read_csv(csv_path)
print(df.head(3))

def combine_info(row):
    # 각 컬럼에 대해 "컬럼명: 값" 형식의 문자열을 생성하여 줄바꿈(\n)으로 연결
    info_lines = [f"{col}: {row[col]}" for col in row.index]
    return "\n".join(info_lines)


a=combine_info(1)
print(a)

















#client = OpenAI()

#client.embeddings.create(
#    model = "text-embedding-ada-002"
#
#)

# def get_embedding(text: str, model: str = "text-embedding-ada-002") -> list:
#     """
#     주어진 텍스트를 OpenAI 임베딩 API를 사용하여 임베딩 벡터로 변환합니다.
    
#     Args:
#         text (str): 임베딩할 텍스트.
#         model (str): 사용할 임베딩 모델 (기본값: "text-embedding-ada-002").
        
#     Returns:
#         list: 임베딩 벡터.
#     """
#     response = openai.Embedding.create(
#         input=text,
#         model=model
#     )
#     return response['data'][0]['embedding']