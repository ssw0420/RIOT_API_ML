import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# 챔피언 특성 데이터 로드
with open('updated_processed_champion_features.json', 'r', encoding='utf-8') as f:
    champion_features = json.load(f)

# 소환사별 상위 챔피언 ID 로드
with open('champion_mastery_top.json', 'r', encoding='utf-8') as f:
    top_champion_mastery_per_summoner = json.load(f)

# 챔피언 특성 데이터프레임 생성
champion_features_df = pd.DataFrame.from_dict(champion_features, orient='index')
champion_features_df.index.name = 'championId'
champion_features_df.index = champion_features_df.index.astype(str)

# 데이터프레임 확인
print("\n챔피언 특성 데이터프레임:")
print(champion_features_df.head())


# 소환사별 특성 데이터를 저장할 딕셔너리
summoner_features = {}

for puuid, champion_mastery_list in top_champion_mastery_per_summoner.items():
    champion_ids = []
    mastery_scores = []
    
    for champ_info in champion_mastery_list:
        champion_ids.append(str(champ_info['championId']))
        mastery_scores.append(float(champ_info['championPoints']))
    
    # 숙련도 점수 로그 변환
    mastery_scores_log = np.log1p(mastery_scores)
    
    # 숙련도 점수 정규화
    scaler = MinMaxScaler()
    mastery_scores_normalized = scaler.fit_transform(np.array(mastery_scores_log).reshape(-1, 1)).flatten()
    
    # 가중치 계산 (합이 1이 되도록)
    weights = mastery_scores_normalized / mastery_scores_normalized.sum()
    
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

# 소환사 특성 데이터프레임 생성
summoner_features_df = pd.DataFrame.from_dict(summoner_features, orient='index')

# 결측치 처리
summoner_features_df.dropna(inplace=True)

# 수치형 특성 선택 및 정규화
numeric_columns = summoner_features_df.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
summoner_features_scaled = scaler.fit_transform(summoner_features_df[numeric_columns])
summoner_features_scaled_df = pd.DataFrame(summoner_features_scaled, index=summoner_features_df.index, columns=numeric_columns)

# 1. 로그 변환된 숙련도 점수 확인
print("\n원본 숙련도 점수:")
print(mastery_scores)
print("\n로그 변환된 숙련도 점수:")
print(mastery_scores_log)

# 2. 정규화된 점수 확인
print("\n정규화된 숙련도 점수:")
print(mastery_scores_normalized)

# 3. 가중치 합계 확인
print("\n가중치 (합계는 1이어야 함):")
print(weights)
print("가중치 합계:", weights.sum())

# 4. 가중치 적용 후 챔피언 특성 합산 확인
print("\n가중치가 적용된 챔피언 특성 (weighted_features):")
print(weighted_features)

# 5. 최종 소환사 특성 데이터프레임 확인
print("\n소환사별 특성 데이터프레임:")
print(summoner_features_df.head())

