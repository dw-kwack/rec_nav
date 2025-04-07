import pandas as pd
import json
import os

def ask_destination(csv_path: str) -> str:
    """
    CSV 파일의 'name' 컬럼에 있는 이름 목록과 비교하여 사용자가 올바른 목적지를 입력할 때까지 반복 입력 받고,
    맞으면 해당 값을 반환합니다.
    """
    df = pd.read_csv(csv_path)
    # CSV 파일의 'name' 컬럼에서 이름 목록 추출
    destination_names = df['name'].tolist()
    
    while True:
        user_input = input("목적지를 입력하세요: ")
        if user_input in destination_names:
            destination = user_input
            print(f"목적지 '{destination}'가 확인되었습니다.")
            return destination
        else:
            print("입력하신 목적지가 CSV에 없습니다. 다시 입력해주세요.")

def load_shared_data(filename="shared_data.json") -> dict:
    """
    공유 데이터 파일(shared_data.json)이 있으면 로드하고, 없으면 기본 딕셔너리를 반환합니다.
    """
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {}

def save_shared_data(data: dict, filename="shared_data.json"):
    """
    data 딕셔너리를 공유 데이터 파일(shared_data.json)에 저장합니다.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # CSV 파일 경로 (실제 경로에 맞게 수정)
    csv_path = '/home/usr/Desktop/recnav/gpt/item_concat.csv'
    destination = ask_destination(csv_path)
    
    # 공유 데이터 파일에서 기존 데이터를 불러오거나 새 딕셔너리 생성
    shared_data = load_shared_data()
    shared_data["destination"] = destination
    
    # 변경된 데이터를 공유 데이터 파일에 저장
    save_shared_data(shared_data)
    
    print("선택한 목적지가 shared_data.json에 저장되었습니다.")
