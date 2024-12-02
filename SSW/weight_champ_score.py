import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 챔피언 특성 데이터
with open('updated_processed_champion_features.json', 'r', encoding='utf-8') as f:
    champion_features = json.load(f)

# 소환사별 상위 3개 챔피언 숙련도 데이터 
with open('champion_mastery_top.json', 'r', encoding='utf-8') as f:
    top_champion_mastery_per_summoner = json.load(f)

# 챔피언 특성 데이터프레임
champion_features_df = pd.DataFrame.from_dict(champion_features, orient='index')
champion_features_df.index.name = 'championId'
champion_features_df.index = champion_features_df.index.astype(str)

# 소환사별 특성 데이터를 저장할 딕셔너리
summoner_features = {}

# 플레이어별로 숙련도 점수를 스케일링하고 가중치를 계산하여 특성 벡터 생성
for puuid, champion_mastery_list in top_champion_mastery_per_summoner.items():
    champion_ids = []
    mastery_scores = []
    
    for champ_info in champion_mastery_list:
        champion_ids.append(str(champ_info['championId']))
        mastery_scores.append(float(champ_info['championPoints']))
    
    # # 숙련도 점수가 없는 경우 건너뜀
    # if len(mastery_scores) == 0:
    #     continue
    
    # 숙련도 점수 로그 변환
    mastery_scores_log = np.log1p(mastery_scores)
    
    # 가중치 계산 (합이 1이 되도록 정규화)
    weights = mastery_scores_log / mastery_scores_log.sum()
    
    # 가중치 계산 (합이 1이 되도록 정규화)
    if mastery_scores_scaled.sum() == 0:
        # 모든 값이 0인 경우 동일한 가중치 부여
        weights = np.ones_like(mastery_scores_scaled) / len(mastery_scores_scaled)
    else:
        weights = mastery_scores_scaled / mastery_scores_scaled.sum()
    
    # 챔피언 특성 가져오기
    try:
        champ_features = champion_features_df.loc[champion_ids]
    except KeyError as e:
        print(f"챔피언 ID {e}에 대한 데이터가 없습니다.")
        continue
    
    # 가중치 적용하여 특성 합산
    weighted_features = champ_features.mul(weights, axis=0)
    summoner_feature_vector = weighted_features.sum()
    
    # 플레이어별 스케일링된 숙련도와 가중치를 저장 (선택 사항)
    mastery_data = pd.DataFrame({
        'championId': champion_ids,
        'championPoints': mastery_scores,
        'masteryLog': mastery_scores_log,
        'masteryScaled': mastery_scores_scaled,
        'weight': weights
    })
    
    # 소환사별로 파일 저장
    # mastery_data.to_csv(f'{puuid}_mastery_weights.csv', index=False)
    
    # 소환사별 특성 저장
    summoner_features[puuid] = summoner_feature_vector

# 소환사 특성 데이터프레임 생성
summoner_features_df = pd.DataFrame.from_dict(summoner_features, orient='index')

# 결과를 CSV 파일로 저장
summoner_features_df.to_csv('summoner_weighted_features.csv')

# 결과 확인
print("소환사별 가중치가 적용된 특성 벡터:")
print(summoner_features_df.head())

# 모든 플레이어의 스케일링된 숙련도와 가중치를 저장
all_mastery_data = {}

for puuid, champion_mastery_list in top_champion_mastery_per_summoner.items():
    champion_ids = []
    mastery_scores = []
    
    for champ_info in champion_mastery_list:
        champion_ids.append(str(champ_info['championId']))
        mastery_scores.append(float(champ_info['championPoints']))
    
    if len(mastery_scores) == 0:
        continue
    
    # 숙련도 점수 로그 변환
    mastery_scores_log = np.log1p(mastery_scores)
    # Min-Max 스케일링
    scaler = MinMaxScaler()
    mastery_scores_scaled = scaler.fit_transform(np.array(mastery_scores_log).reshape(-1,1)).flatten()
    # 가중치 계산
    if mastery_scores_scaled.sum() == 0:
        weights = np.ones_like(mastery_scores_scaled) / len(mastery_scores_scaled)
    else:
        weights = mastery_scores_scaled / mastery_scores_scaled.sum()
    
    # 데이터 저장
    mastery_data = pd.DataFrame({
        'championId': champion_ids,
        'championPoints': mastery_scores,
        'masteryLog': mastery_scores_log,
        'masteryScaled': mastery_scores_scaled,
        'weight': weights
    })
    
    # 딕셔너리에 저장
    all_mastery_data[puuid] = mastery_data.to_dict(orient='list')

# 모든 플레이어의 숙련도 데이터 저장
with open('player_mastery_weights.json', 'w', encoding='utf-8') as f:
    json.dump(all_mastery_data, f, ensure_ascii=False, indent=4)

print("플레이어별 스케일링된 숙련도와 가중치 = 'player_mastery_weights.json'")
