import json


with open('processed_champion_features.json', 'r', encoding='utf-8') as f:
    champion_features = json.load(f)


for champ_id, features in champion_features.items():
    features.pop('id', None)   # 'id' 필드 제거
    features.pop('name', None) # 'name' 필드 제거

with open('updated_processed_champion_features.json', 'w', encoding='utf-8') as f:
    json.dump(champion_features, f, ensure_ascii=False, indent=4)

print("'id'와 'name' 필드가 제거된 JSON 파일을 'updated_processed_champion_features.json'에 저장")