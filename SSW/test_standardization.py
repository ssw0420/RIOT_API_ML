import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드
with open('updated_processed_champion_features.json', 'r', encoding='utf-8') as f:
    champion_features = json.load(f)

with open('champion_mastery_top.json', 'r', encoding='utf-8') as f:
    top_champion_mastery_per_summoner = json.load(f)

# 챔피언 특성 데이터프레임 생성
champion_features_df = pd.DataFrame.from_dict(champion_features, orient='index')
champion_features_df.index.name = 'championId'
champion_features_df.index = champion_features_df.index.astype(str)

# 2. 소환사별 가중치 계산 및 확인
summoner_weights = {}

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

    # 표준화 적용
    scaler = StandardScaler()
    mastery_scores_standardized = scaler.fit_transform(mastery_scores_log).flatten()

    # 최소값을 0으로 이동시키기
    min_value = mastery_scores_standardized.min()
    mastery_scores_shifted = mastery_scores_standardized - min_value

    # 가중치 계산 (합이 1이 되도록 정규화)
    if mastery_scores_shifted.sum() == 0:
        weights = np.ones_like(mastery_scores_shifted) / len(mastery_scores_shifted)
    else:
        weights = mastery_scores_shifted / mastery_scores_shifted.sum()

    # 가중치 확인
    print(f"\n플레이어 {puuid}의 가중치:")
    for cid, score, w in zip(champion_ids, mastery_scores, weights):
        print(f"챔피언 ID: {cid}, 숙련도 점수: {score}, 가중치: {w:.4f}")

    # 가중치를 딕셔너리에 저장
    summoner_weights[puuid] = {
        'championIds': champion_ids,
        'masteryScores': mastery_scores,
        'weights': weights.tolist()
    }

# 3. 가중치 결과를 파일로 저장
with open('standardization_summoner_weights.json', 'w', encoding='utf-8') as f:
    json.dump(summoner_weights, f, ensure_ascii=False, indent=4)

# 4. 소환사별 특성 벡터 생성
summoner_features = {}

for puuid, weights_info in summoner_weights.items():
    champion_ids = weights_info['championIds']
    weights = np.array(weights_info['weights'])

    # 챔피언 특성 가져오기
    try:
        champ_features = champion_features_df.loc[champion_ids]
    except KeyError as e:
        print(f"챔피언 ID {e}에 대한 데이터가 없습니다.")
        continue

    # 가중치 적용하여 특성 합산
    weighted_features = champ_features.mul(weights, axis=0)
    summoner_feature_vector = weighted_features.sum()

    # 소환사별 특성 저장
    summoner_features[puuid] = summoner_feature_vector

# 5. 소환사 특성 데이터프레임 생성
summoner_features_df = pd.DataFrame.from_dict(summoner_features, orient='index')

# 6. 결과를 CSV 파일로 저장
summoner_features_df.to_csv('standardization_summoner_weighted_features.csv')

# 7. 결과 확인
print("\n소환사별 가중치가 적용된 특성 벡터:")
print(summoner_features_df.head())
