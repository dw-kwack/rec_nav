import pandas as pd

def ask_destination(csv_path: str) -> str:
    """
    CSV 파일의 'name' 컬럼에 있는 이름 목록과 비교하여 사용자가 올바른 목적지를 입력할 때까지 반복 입력 받고,
    맞으면 해당 값을 반환.
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

if __name__ == "__main__":
    # CSV 파일 경로 (실제 경로에 맞게 수정)
    csv_path = '/home/usr/Desktop/recnav/gpt/item_concat.csv'
    destination = ask_destination(csv_path)
    #print(destination)