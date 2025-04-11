# caption.py (BLIP 모델 기반)
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import json

# 모델과 프로세서 로드
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 글로벌 변수: 캡셔닝 결과에서 'what' 필드 값들을 저장 (예: ["The Starry Night", "Vincent van Gogh"])
user_info = []

def load_shared_data(filename="shared_data.json"):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {"user_info": [], "direct": [], "indirect": []}

def save_shared_data(data, filename="shared_data.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 이미지 경로 (실제 경로에 맞게 수정)
image_path = "/home/usr/Desktop/recnav/gpt/test.jpg"
image = Image.open(image_path).convert("RGB")

# BLIP 모델에 전달할 프롬프트 텍스트 구성  
prompt_text = (
    "Output must follow exactly this JSON format:\n"
    '{"captions": [{"who": "string", "what": "string"}]}\n'
    "The image shows two people in a gallery looking at a painting on the wall. "
    "The painting is 'The Starry Night' by Vincent van Gogh. "
    "Ensure that in the 'what' field, you include either 'The Starry Night' or 'Vincent van Gogh' "
    "without including any additional action phrases like 'looking at'. "
    "Output only valid JSON without any additional text."
)

# BLIP 모델을 이용해 캡셔닝 진행 (입력: 이미지와 프롬프트)
inputs = processor(image, prompt_text, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=200)
caption = processor.decode(out[0], skip_special_tokens=True).strip()

print("Structured caption:")
print(caption)

# JSON 형식으로 파싱 시도
try:
    data = json.loads(caption)
    # 각 캡션에서 "what" 필드를 추출하여 user_info 리스트에 저장
    user_info = [item["what"] for item in data.get("captions", [])]
except json.JSONDecodeError:
    print("Response is not valid JSON:")
    print(caption)
    data = {}

# shared_data.json 파일 업데이트
shared_data = load_shared_data()
shared_data["user_info"] = user_info
save_shared_data(shared_data)
