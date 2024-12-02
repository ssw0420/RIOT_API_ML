import json

# 1. JSON 파일에서 데이터 로드
with open('all_summoner_data.json', 'r', encoding='utf-8') as f:
    summoner_data = json.load(f)

# 2. 'puuid' 값 추출
puuid_list = [entry['puuid'] for entry in summoner_data if 'puuid' in entry]

# 3. 'puuid' 리스트를 새로운 JSON 파일에 저장
with open('puuid_list.json', 'w', encoding='utf-8') as f:
    json.dump(puuid_list, f, ensure_ascii=False, indent=4)

print(f"총 {len(puuid_list)}개의 'puuid'를 'puuid_list.json' 파일에 저장했습니다.")
