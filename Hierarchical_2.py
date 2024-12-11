import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os


### [STEP 1] 챔피언 특성 및 플레이어별 가중치 데이터 로드
# - updated_processed_champion_features.json: 챔피언별 특성 정보
# - scale_summoner_weights.json: 플레이어별 상위 3개 챔피언의 숙련도 기반 가중치 정보
with open('SSW/updated_processed_champion_features.json', 'r', encoding='utf-8') as f:
    champion_features = json.load(f)

with open('SSW/scale_summoner_weights.json', 'r', encoding='utf-8') as f:
    summoner_weights = json.load(f)


### [STEP 2] 챔피언 특성 데이터프레임 생성
# - champion_features_dict를 DataFrame으로 변환
champion_features_df = pd.DataFrame.from_dict(champion_features, orient='index')
champion_features_df.index.name = 'championId'
champion_features_df.index = champion_features_df.index.astype(str)


### [STEP 3] 소환사별 특성 벡터 생성 및 챔피언별 가중치 적용 특성 저장
# - summoner_features: 소환사별 종합 특성 벡터(상위 3챔 가중치 합)
# - summoner_champion_features: 소환사별 챔피언별 가중치 적용 특성
summoner_features = {}
summoner_champion_features = {}

for puuid, weights_info in summoner_weights.items():
    champion_ids = weights_info['championIds']
    weights = np.array(weights_info['weights'])
    
    # 챔피언 특성 가져오기
    try:
        champ_features = champion_features_df.loc[champion_ids]
    except KeyError as e:
        print(f"챔피언 ID {e}에 대한 데이터가 없습니다.")
        continue
    
    # 가중치 적용하여 특성 합산 (플레이어 숙련도 기반)
    weighted_features = champ_features.mul(weights, axis=0)
    summoner_feature_vector = weighted_features.sum()
    summoner_features[puuid] = summoner_feature_vector

    # 소환사별 챔피언별 특성 (추후 분석용)
    summoner_champion_features[puuid] = {}
    for idx, cid in enumerate(champion_ids):
        w = weights[idx]
        features = champ_features.loc[cid]
        weighted_feature = features * w
        summoner_champion_features[puuid][cid] = weighted_feature.to_dict()


### [STEP 4] 소환사 특성 DataFrame 생성 및 전처리
# - 결측치 제거(필요시)
# - 표준화(Scaling) 진행
summoner_features_df = pd.DataFrame.from_dict(summoner_features, orient='index')
summoner_features_df = summoner_features_df.dropna()  # 결측치 제거

scaler = StandardScaler()
summoner_features_scaled = scaler.fit_transform(summoner_features_df)


### [STEP 5] 계층적 군집화 수행 (Linkage 계산)
# - 유클리드 거리, 워드 연결법 사용
linked = linkage(summoner_features_scaled, method='ward')


### [STEP 6] 덴드로그램 시각화
# - truncate_mode='lastp', p=30을 사용해 상위 30개 클러스터만 표시 (복잡성 완화)
plt.figure(figsize=(10, 7))
dendrogram(linked, truncate_mode='lastp', p=30)
plt.title('Dendrogram')
plt.xlabel('Player')
plt.ylabel('Linkage')
plt.show()



### [STEP 7] 거리 임계값 설정을 통한 클러스터 결정
# - t=40으로 설정 (덴드로그램 해석 결과 예시값)
cluster_labels = fcluster(linked, t=40, criterion='distance')
summoner_features_df['cluster'] = cluster_labels



### [STEP 8] 클러스터 결과 저장 (JSON, CSV)
# - 플레이어별 클러스터 정보
summoner_cluster_results = summoner_features_df['cluster'].to_dict()
with open('SSW/Hierarchical_2/hierarchical_summoner_cluster_results.json', 'w', encoding='utf-8') as f:
    json.dump(summoner_cluster_results, f, ensure_ascii=False, indent=4)

# - 클러스터별 평균 특성 계산 및 저장
cluster_centers = summoner_features_df.groupby('cluster').mean()
cluster_centers_dict = cluster_centers.to_dict(orient='index')
with open('SSW/Hierarchical_2/hierarchical_cluster_centers.json', 'w', encoding='utf-8') as f:
    json.dump(cluster_centers_dict, f, ensure_ascii=False, indent=4)

# - 소환사별 챔피언별 가중치 적용 특성 저장
with open('SSW/Hierarchical_2/hierarchical_summoner_champion_weighted_features.json', 'w', encoding='utf-8') as f:
    json.dump(summoner_champion_features, f, ensure_ascii=False, indent=4)

# 클러스터별 플레이어 수 출력
cluster_counts = summoner_features_df['cluster'].value_counts()
print("\n클러스터별 플레이어 수:")
print(cluster_counts)

# 결과를 CSV로 저장
summoner_features_df.to_csv('SSW/Hierarchical_2/hierarchical_summoner_cluster_results.csv')
cluster_centers.to_csv('SSW/Hierarchical_2/hierarchical_cluster_centers.csv')


### [STEP 9] t-SNE를 사용한 2D 시각화
# - n_components=2: 2차원 변환
# - perplexity=30, n_iter=1000 예시값
# - random_state=42로 결과 재현성 확보
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
summoner_features_2d = tsne.fit_transform(summoner_features_scaled)

# t-SNE 결과 시각화
plt.figure(figsize=(10, 7))
plt.scatter(summoner_features_2d[:, 0], summoner_features_2d[:, 1], c=cluster_labels, cmap='tab10')
plt.title('t-SNE')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.colorbar(label='cluster')
plt.show()


### [STEP 10] t-SNE 3D 시각화
# - 3차원으로 시각화 (n_components=3)
from mpl_toolkits.mplot3d import Axes3D

tsne_3d = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
summoner_features_3d = tsne_3d.fit_transform(summoner_features_scaled)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(summoner_features_3d[:,0], summoner_features_3d[:,1], summoner_features_3d[:,2], c=cluster_labels, cmap='tab10')
plt.title('t-SNE 3D Visualization')
ax.set_xlabel('Dim1')
ax.set_ylabel('Dim2')
ax.set_zlabel('Dim3')
plt.colorbar(sc, label='cluster')
plt.show()


### [STEP 11] PCA 시각화 (2D 예시)
pca_2d = PCA(n_components=2)
summoner_features_pca_2d = pca_2d.fit_transform(summoner_features_scaled)

plt.figure(figsize=(10,7))
plt.scatter(summoner_features_pca_2d[:, 0], summoner_features_pca_2d[:, 1], c=cluster_labels, cmap='tab10')
plt.title('PCA 2D Visualization')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(label='cluster')
plt.show()


### [STEP 12] PCA 시각화 (3D 예시)
pca_3d = PCA(n_components=3)
summoner_features_pca_3d = pca_3d.fit_transform(summoner_features_scaled)
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(summoner_features_pca_3d[:,0], summoner_features_pca_3d[:,1], summoner_features_pca_3d[:,2], c=cluster_labels, cmap='tab10')
plt.title('PCA 3D Visualization')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.colorbar(sc, label='cluster')
plt.show()


