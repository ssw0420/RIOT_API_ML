import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# 1. 챔피언 특성 데이터 로드
with open('updated_processed_champion_features.json', 'r', encoding='utf-8') as f:
    champion_features = json.load(f)

# 2. 소환사별 상위 챔피언 ID 로드
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

# K-평균 클러스터링 및 엘보우 방법 적용
sse = []
k_list = range(1, 20)

for k in k_list:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(summoner_features_scaled_df)
    sse.append(kmeans.inertia_)

# 결과 시각화
plt.figure(figsize=(8, 5))
plt.plot(k_list, sse, marker='o')
plt.xlabel('클러스터 수 (k)')
plt.ylabel('SSE')
plt.title('엘보우 방법을 사용한 최적의 k 찾기')
plt.show()

#########################################
k_optimal = 5

# K-Means 모델 학습
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
kmeans.fit(summoner_features_scaled_df)

# 클러스터 레이블 저장
summoner_features_df['cluster'] = kmeans.labels_

# 결과 확인
print("\n클러스터 할당 결과:")
print(summoner_features_df[['cluster']].head())


# 결과를 CSV 파일로 저장
summoner_features_df.to_csv('weighted_summoner_clustering_results.csv')

# 또는 JSON 파일로 저장
summoner_features_df.to_json('weighted_summoner_clustering_results.json', orient='index', indent=4)

print("\n클러스터링 결과를 저장했습니다.")


# 클러스터별 평균 특성 계산
cluster_centers = summoner_features_df.groupby('cluster').mean()

print("\n클러스터별 평균 특성:")
print(cluster_centers)


# PCA로 3차원으로 축소
pca = PCA(n_components=3)
principal_components = pca.fit_transform(summoner_features_scaled_df)

# 결과를 데이터프레임으로 생성
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'], index=summoner_features_df.index)
pca_df['cluster'] = summoner_features_df['cluster']

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=pca_df['cluster'], cmap='Set2')
ax.set_title('PCA 3D 시각화')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.legend(*scatter.legend_elements(), title="클러스터")
plt.show()
