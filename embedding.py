from openai import OpenAI
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

def get_embedding(text: str) -> list:
    """
    최신 OpenAI 임베딩 API를 사용하여 텍스트의 임베딩 벡터를 반환합니다.
    """
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
        encoding_format="float"
    )
    return response.data[0].embedding

def load_csv_and_generate_embeddings(csv_path: str) -> pd.DataFrame:
    """
    CSV 파일을 로드하고, coordinate 컬럼은 제외한 나머지 정보를 결합하여 임베딩을 생성합니다.
    - 결합 대상: name, information, region, writter
    - 카테고리 컬럼(값이 1인 경우만)을 결합하여 포함 (painting, potery, royal, impressionist, furniture,
      religion, buda, shamanism, Christianus, korea, hindu, egyptian, natural history, fossil, dinosaur, animal, mineral)
    """
    df = pd.read_csv(csv_path)
    print("CSV 파일의 컬럼:", df.columns.tolist())
    
    category_cols = [
        "painting", "potery", "royal", "impressionist", "furniture", 
        "religion", "buda", "shamanism", "Christianus", "korea", 
        "hindu", "egyptian", "natural history", "fossil", "dinosaur", 
        "animal", "mineral"
    ]
    
    def combine_info(row):
        base_info = (
            f"name: {row['name']}\n"
            f"information: {row['information']}\n"
            f"region: {row['region']}\n"
            f"writter: {row['writter']}\n"
        )
        active_categories = [cat for cat in category_cols if row[cat] == 1]
        if active_categories:
            base_info += "categories: " + ", ".join(active_categories)
        return base_info
    
    df["combined_text"] = df.apply(combine_info, axis=1)
    df["embedding"] = df["combined_text"].apply(get_embedding)
    
    return df










