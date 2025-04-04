from pydantic import BaseModel
from typing import List
from openai import OpenAI
import base64
import os
from PIL import Image
import numpy as np
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from langchain_openai import OpenAIEmbeddings

openai_api_key = os.getenv('OPENAI_API_KEY')

class ImageCaption(BaseModel):
    who: str   # 예: "man", "woman", "child" 등
    sex: str   # 예: "male", "female"
    what: str  # 예: "The Starry Night"

class ImageCaptionExtraction(BaseModel):
    captions: List[ImageCaption]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    

def get_embedding(text,engine):
    text = text.replace()
    
client  = OpenAI()
    
image_path = "/home/usr/Desktop/recnav/gpt/test.jpg"

base64_image = encode_image(image_path)


completion = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "당신은 구조화된 데이터 추출 전문가입니다. 이미지에 대한 설명을 받아, 주어진 구조에 맞게 JSON 형식으로 변환하세요."
        },
        {
            "role": "user",
            "content": (
                "이미지 설명: 두 사람이 박물관 또는 갤러리에서 벽에 걸린 액자 속의 'The Starry Night' 작품을 바라보고 있습니다. "
                "한 명은 남성(남자)이고, 다른 한 명은 여성(여자)으로 보입니다. "
                "출력은 각 객체가 'who', 'sex', 'what' 키를 포함하는 JSON 배열 형식이어야 합니다."
            )
        }
    ],
    response_format=ImageCaptionExtraction,
)

print(completion.choices[0].message.content)