import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# 1. 챔피언 특성 데이터 로드
with open('updated_processed_champion_features.json', 'r', encoding='utf-8') as f:
    champion_features = json.load(f)

# 2. 소환사별 상위 챔피언 ID 로드
with open('top_champions_per_summoner.json', 'r', encoding='utf-8') as f:
    top_champions_per_summoner = json.load(f)

# 챔피언 특성 데이터프레임 생성
champion_features_df = pd.DataFrame.from_dict(champion_features, orient='index')

# 인덱스를 챔피언 ID로 설정
champion_features_df.index.name = 'championId'

# 데이터프레임 확인
print("\n챔피언 특성 데이터프레임:")
print(champion_features_df.head())


# 소환사별 특성 데이터를 저장할 딕셔너리
summoner_features = {}

for puuid, champion_ids in top_champions_per_summoner.items():
    champion_ids = [str(champ_id) for champ_id in champion_ids]

    
    # 챔피언 특성 데이터프레임에서 해당 챔피언의 특성 선택
    try:
        champ_features = champion_features_df.loc[champion_ids]
    except KeyError as e:
        print(f"챔피언 ID {e}에 대한 데이터가 없습니다.")
        continue
    
    # 특성 합산 또는 평균 계산
    total_features = champ_features.sum()
    average_features = champ_features.mean()
    
    # 소환사별 특성 저장
    summoner_features[puuid] = average_features


# 소환사 특성 데이터프레임 생성
summoner_features_df = pd.DataFrame.from_dict(summoner_features, orient='index')

# 데이터프레임 확인
print("\n소환사 특성 데이터프레임:")
print(summoner_features_df.head())


# 결측치 확인
print("\n결측치 개수:")
print(summoner_features_df.isnull().sum())


print("챔피언 특성 데이터프레임 크기:", champion_features_df.shape)
print("소환사 특성 데이터프레임 크기:", summoner_features_df.shape)

# 결측치가 있는 행 제거 (필요한 경우)
summoner_features_df.dropna(inplace=True)


# 수치형 특성 선택 (필요한 경우)
numeric_columns = summoner_features_df.select_dtypes(include=[np.number]).columns

# 데이터 정규화
scaler = StandardScaler()
summoner_features_scaled = scaler.fit_transform(summoner_features_df[numeric_columns])

# 정규화된 데이터프레임 생성
summoner_features_scaled_df = pd.DataFrame(summoner_features_scaled, index=summoner_features_df.index, columns=numeric_columns)

sse = []
k_list = range(1, 10)

for k in k_list:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(summoner_features_scaled_df)
    sse.append(kmeans.inertia_)  # inertia_ 속성은 클러스터 내 SSE

# # 결과 시각화
# plt.figure(figsize=(8, 5))
# plt.plot(k_list, sse, marker='o')
# plt.xlabel('(k)')
# plt.ylabel('SSE')
# plt.title('k')
# plt.show()

# 최적의 클러스터 수 설정 (예: k=5)
k_optimal = 4

# K-Means 모델 학습
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
kmeans.fit(summoner_features_scaled_df)

# 클러스터 레이블 저장
summoner_features_df['cluster'] = kmeans.labels_

# 결과 확인
print("\n클러스터 할당 결과:")
print(summoner_features_df[['cluster']].head())


# 결과를 CSV 파일로 저장
summoner_features_df.to_csv('summoner_clustering_results.csv')

# 또는 JSON 파일로 저장
summoner_features_df.to_json('summoner_clustering_results.json', orient='index', indent=4)

print("\n클러스터링 결과를 저장했습니다.")


# 클러스터별 평균 특성 계산
cluster_centers = summoner_features_df.groupby('cluster').mean()

print("\n클러스터별 평균 특성:")
print(cluster_centers)


# PCA로 2차원으로 축소
pca = PCA(n_components=2)
principal_components = pca.fit_transform(summoner_features_scaled_df)

# 결과를 데이터프레임으로 생성
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=summoner_features_df.index)
pca_df['cluster'] = summoner_features_df['cluster']

# 시각화
plt.figure(figsize=(10, 7))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster', palette='Set2')
plt.title('PCA')
plt.show()

