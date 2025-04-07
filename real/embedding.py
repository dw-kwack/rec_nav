# embedding.py
from openai import OpenAI
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import json
import hashlib

# OpenAI API 키 및 client 초기화
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

CACHE_FILE = "embedding_cache.json"

def load_cache(filename: str = CACHE_FILE) -> dict:
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {}

def save_cache(cache: dict, filename: str = CACHE_FILE):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def hash_text(text: str) -> str:
    """
    주어진 텍스트에 대해 SHA256 해시값을 반환합니다.
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

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
    임베딩은 캐시를 활용하여 이미 계산된 결과가 있으면 재사용합니다.
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
    
    # 캐시 로드
    cache = load_cache()
    
    embeddings = []
    for text in df["combined_text"]:
        key = hash_text(text)
        if key in cache:
            emb = cache[key]
        else:
            emb = get_embedding(text)
            cache[key] = emb
        embeddings.append(emb)
    
    # 캐시 저장 (변경된 결과 업데이트)
    save_cache(cache)
    
    df["embedding"] = embeddings
    return df

if __name__ == "__main__":
    csv_path = '/home/usr/Desktop/recnav/gpt/item_concat.csv'
    df_embedded = load_csv_and_generate_embeddings(csv_path)
    print("임베딩 생성 완료. DataFrame 미리보기:")
    print(df_embedded.head())
    
    # 예시: PCA로 2차원 시각화 (옵션)
    embeddings = df_embedded["embedding"].tolist()
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    df_embedded["pca_x"] = embeddings_2d[:, 0]
    df_embedded["pca_y"] = embeddings_2d[:, 1]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(df_embedded["pca_x"], df_embedded["pca_y"], alpha=0.7)
    for _, row in df_embedded.iterrows():
        plt.text(row["pca_x"], row["pca_y"], str(row["name"]), fontsize=8)
    plt.title("임베딩 공간 (PCA 2D)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.show()
