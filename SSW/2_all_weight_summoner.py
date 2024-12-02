import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

with open('updated_processed_champion_features.json', 'r', encoding='utf-8') as f:
    champion_features = json.load(f)

with open('champion_mastery_top.json', 'r', encoding='utf-8') as f:
    top_champion_mastery_per_summoner = json.load(f)

# 챔피언 특성 데이터프레임 생성
champion_features_df = pd.DataFrame.from_dict(champion_features, orient='index')
champion_features_df.index.name = 'championId'
champion_features_df.index = champion_features_df.index.astype(str)

# 모든 플레이어의 상위 3개 챔피언 숙련도 점수 수집
all_mastery_scores = []

for champion_mastery_list in top_champion_mastery_per_summoner.values():
    for champ_info in champion_mastery_list:
        mastery_score = float(champ_info['championPoints'])
        all_mastery_scores.append(mastery_score)

# 3. 로그 변환 적용
all_mastery_scores_log = np.log1p(all_mastery_scores).reshape(-1, 1)

# 4. 전체 데이터에 대해 Min-Max 스케일링
scaler = MinMaxScaler()
scaler.fit(all_mastery_scores_log)

# 소환사별 가중치 계산
summoner_weights = {}  # 각 플레이어별 가중치를 저장할 딕셔너리

for puuid, champion_mastery_list in top_champion_mastery_per_summoner.items():
    champion_ids = []
    mastery_scores = []
    
    for champ_info in champion_mastery_list:
        champion_ids.append(str(champ_info['championId']))
        mastery_scores.append(float(champ_info['championPoints']))
    
    # 숙련도 점수가 없는 경우 건너뜀
    if len(mastery_scores) == 0:
        continue
    
    # 숙련도 점수 로그 변환
    mastery_scores_log = np.log1p(mastery_scores).reshape(-1, 1)
    
    # 이전에 fit한 scaler를 사용하여 스케일링
    mastery_scores_scaled = scaler.transform(mastery_scores_log).flatten()
    
    # 가중치 계산 (합이 1이 되도록 정규화)
    if mastery_scores_scaled.sum() == 0:
        # 모든 값이 0인 경우 동일한 가중치 부여
        weights = np.ones_like(mastery_scores_scaled) / len(mastery_scores_scaled)
    else:
        weights = mastery_scores_scaled / mastery_scores_scaled.sum()
    
    # 가중치가 극단적인 값인지 확인
    print(f"\n플레이어 {puuid}의 가중치:")
    for cid, score, w in zip(champion_ids, mastery_scores, weights):
        print(f"챔피언 ID: {cid}, 숙련도 점수: {score}, 가중치: {w:.4f}")
    
    # 가중치를 딕셔너리에 저장
    summoner_weights[puuid] = {
        'championIds': champion_ids,
        'masteryScores': mastery_scores,
        'weights': weights.tolist()
    }

#  가중치 결과
with open('2_all_summoner_weights.json', 'w', encoding='utf-8') as f:
    json.dump(summoner_weights, f, ensure_ascii=False, indent=4)
