import json

def extract_id_and_key(input_file: str, output_file: str):
    try:
        # 원본 JSON 데이터 로드
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # "data" 키 내부의 챔피언 정보에 접근
        champions = data.get("data", {})
        
        # id와 key만 추출
        filtered_data = {}
        for champ_name, champ_info in champions.items():
            # 각 챔피언의 id와 key를 추출
            champ_id = champ_info.get("id")
            champ_key = champ_info.get("key")
            
            # 추출한 데이터를 새로운 딕셔너리에 저장
            filtered_data[champ_name] = {
                "id": champ_id,
                "key": champ_key
            }
        
        # 새로운 JSON 파일로 저장
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(filtered_data, file, ensure_ascii=False, indent=4)
        
        print(f"성공적으로 필터링된 JSON 파일이 생성되었습니다: {output_file}")
    
    except FileNotFoundError:
        print(f"에러: 파일을 찾을 수 없습니다 - {input_file}")
    except json.JSONDecodeError:
        print("에러: JSON 파일의 형식이 올바르지 않습니다.")
    except Exception as e:
        print(f"예상치 못한 에러 발생: {e}")

if __name__ == "__main__":
    # 입력 및 출력 파일 경로 설정
    input_json = 'Crawling\ddragon_champion_all_info.json'          # 원본 JSON 파일 경로
    output_json = 'Crawling\champions_filtered.json'  # 출력할 JSON 파일 경로
    
    # 함수 호출
    extract_id_and_key(input_json, output_json)
