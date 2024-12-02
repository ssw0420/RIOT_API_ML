import json

with open('champion_mastery_top.json', 'r', encoding='utf-8') as f:
    mastery_data = json.load(f)

top_champions_per_summoner = {}

for puuid, champion_masteries in mastery_data.items():
    # 각 소환사의 상위 3개 챔피언 ID 추출
    top_champion_ids = [str(entry['championId']) for entry in champion_masteries]
    top_champions_per_summoner[puuid] = top_champion_ids

# 결과를 JSON 파일로 저장
with open('top_champions_per_summoner.json', 'w', encoding='utf-8') as f:
    json.dump(top_champions_per_summoner, f, ensure_ascii=False, indent=4)

print("상위 3개 챔피언 ID를 'top_champions_per_summoner.json' 파일에 저장")
