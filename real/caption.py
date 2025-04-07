# caption.py
from openai import OpenAI
import base64
import os
import json

client = OpenAI()
openai_api_key = os.getenv("OPENAI_API_KEY")

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

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
image_path = "/home/usr/Desktop/recnav/gpt/test.jpg"
base64_image = encode_image(image_path)

prompt = (
    "You are an expert image captioner. Analyze the image and generate a structured caption in JSON format. "
    "The output must follow exactly this format:\n"
    '{"captions": [{"who": "string", "what": "string"}]}\n'
    "The image shows two people in a gallery looking at a painting on the wall. "
    "The painting is 'The Starry Night' by Vincent van Gogh. "
    "Ensure that in the 'what' field, you include either 'The Starry Night' or 'Vincent van Gogh' (or both if needed) "
    "without including any additional action phrases like 'looking at'. "
    "Output only valid JSON without any additional text."
)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are an expert image captioner."},
        {"role": "user", "content": prompt},
        {"role": "user", "content": [
            {"type": "text", "text": "Caption the image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]}
    ],
    temperature=0.7,
    max_tokens=200
)

output = completion.choices[0].message.content.strip()

def clean_json_output(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text

cleaned_output = clean_json_output(output)

try:
    data = json.loads(cleaned_output)
    print("Structured caption:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    # 업데이트: 각 캡션에서 "what" 필드 추출
    user_info = [caption["what"] for caption in data.get("captions", [])]
except json.JSONDecodeError:
    print("Response is not valid JSON:")
    print(cleaned_output)

# shared_data.json 파일 업데이트
shared_data = load_shared_data()
shared_data["user_info"] = user_info
save_shared_data(shared_data)
