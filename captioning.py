from openai import OpenAI
import base64 # 이미지 인코딩
import os # bash api 불러오기
from PIL import Image 
import numpy as np
import json



client  = OpenAI()

openai_api_key = os.getenv('OPENAI_API_KEY')

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
image_path = "/home/usr/Desktop/recnav/gpt/test.jpg"

base64_image = encode_image(image_path)


completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                { "type": "text", "text": "what's in this image?" },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
)

print(completion.choices[0].message.content)