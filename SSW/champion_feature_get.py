import json

with open('ddragon_champion_all_info.json', 'r', encoding='utf-8') as f:
    champion_data = json.load(f)


champion_features = {}
all_tags = ['Fighter', 'Mage', 'Assassin', 'Marksman', 'Tank', 'Support']

# 제외할 'stats' 필드 목록
exclude_stats_fields = ['mp', 'mpperlevel', 'mpregen', 'mpregenperlevel']

for champion_name, champion_details in champion_data['data'].items():
    # 식별자 추출
    champ_id = champion_details.get('key')
    name = champion_details.get('name')
    
    # info 필드 추출
    info = champion_details.get('info', {})
    attack = info.get('attack', 0)
    defense = info.get('defense', 0)
    magic = info.get('magic', 0)
    difficulty = info.get('difficulty', 0)
    
    # tags 필드 추출 및 원-핫 인코딩
    tags = champion_details.get('tags', [])
    tags_one_hot = {tag: 1 if tag in tags else 0 for tag in all_tags}
    
    # stats 필드 추출 (제외할 필드 제외)
    stats = champion_details.get('stats', {})
    stats_filtered = {k: v for k, v in stats.items() if k not in exclude_stats_fields}
    
    # 챔피언의 모든 특성 합치기
    champion_feature = {
        'id': champ_id,
        'name': name,
        'attack': attack,
        'defense': defense,
        'magic': magic,
        'difficulty': difficulty,
        **tags_one_hot,
        **stats_filtered
    }
    
    # 결과 딕셔너리에 저장
    champion_features[champ_id] = champion_feature

# 결과를 JSON 파일로 저장
with open('processed_champion_features.json', 'w', encoding='utf-8') as f:
    json.dump(champion_features, f, ensure_ascii=False, indent=4)

print("챔피언 특성 데이터 'processed_champion_features.json' 파일에 저장")
