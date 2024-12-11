# import json
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import AgglomerativeClustering
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# import os
# from sklearn.metrics import silhouette_score  # 실루엣 계수 계산 추가

# ### [STEP 1] 챔피언 특성 및 플레이어별 가중치 데이터 로드
# with open('SSW/updated_processed_champion_features.json', 'r', encoding='utf-8') as f:
#     champion_features = json.load(f)

# with open('SSW/scale_summoner_weights.json', 'r', encoding='utf-8') as f:
#     summoner_weights = json.load(f)

# # 결과 디렉토리
# output_dir = 'SSW/New_Hierarchical'
# os.makedirs(output_dir, exist_ok=True)

# ### [STEP 2] 챔피언 특성 DataFrame 생성
# champion_features_df = pd.DataFrame.from_dict(champion_features, orient='index')
# champion_features_df.index.name = 'championId'
# champion_features_df.index = champion_features_df.index.astype(str)

# ### [STEP 3] 소환사별 특성 벡터 생성
# summoner_features = {}
# summoner_champion_features = {}

# for puuid, weights_info in summoner_weights.items():
#     champion_ids = weights_info['championIds']
#     weights = np.array(weights_info['weights'])
    
#     try:
#         champ_features = champion_features_df.loc[champion_ids]
#     except KeyError as e:
#         print(f"챔피언 ID {e} 데이터 없음")
#         continue

#     weighted_features = champ_features.mul(weights, axis=0)
#     summoner_feature_vector = weighted_features.sum()
#     summoner_features[puuid] = summoner_feature_vector

#     summoner_champion_features[puuid] = {}
#     for idx, cid in enumerate(champion_ids):
#         w = weights[idx]
#         features = champ_features.loc[cid]
#         weighted_feature = features * w
#         summoner_champion_features[puuid][cid] = weighted_feature.to_dict()

# ### [STEP 4] 전처리 (스케일링)
# summoner_features_df = pd.DataFrame.from_dict(summoner_features, orient='index')
# summoner_features_df = summoner_features_df.dropna()

# scaler = StandardScaler()
# summoner_features_scaled = scaler.fit_transform(summoner_features_df)

# ### [STEP 5] 계층적 군집화 - Average Linkage + Cosine 거리
# n_clusters = 6 # 클러스터 개수 지정
# cluster = AgglomerativeClustering(
#     n_clusters=n_clusters,
#     metric='cosine',
#     linkage='average', 
#     compute_full_tree='auto'
# )

# cluster_labels = cluster.fit_predict(summoner_features_scaled)
# summoner_features_df['cluster'] = cluster_labels

# ### [STEP 5.1] 실루엣 계수 계산 추가
# sil_score = silhouette_score(summoner_features_scaled, cluster_labels, metric='cosine')
# print(f"Silhouette Score (Average+Cosine): {sil_score:.4f}")
# with open(os.path.join(output_dir, 'silhouette_score.txt'), 'w', encoding='utf-8') as f:
#     f.write(f"Silhouette Score (Average+Cosine): {sil_score:.4f}\n")

# ### [STEP 6] 결과 저장
# summoner_cluster_results = summoner_features_df['cluster'].to_dict()
# with open(os.path.join(output_dir, 'hierarchical_cosine_average_cluster_results.json'), 'w', encoding='utf-8') as f:
#     json.dump(summoner_cluster_results, f, ensure_ascii=False, indent=4)

# cluster_centers = summoner_features_df.groupby('cluster').mean()
# cluster_centers_dict = cluster_centers.to_dict(orient='index')
# with open(os.path.join(output_dir, 'hierarchical_cosine_average_cluster_centers.json'), 'w', encoding='utf-8') as f:
#     json.dump(cluster_centers_dict, f, ensure_ascii=False, indent=4)

# with open(os.path.join(output_dir, 'hierarchical_cosine_average_summoner_champion_weighted_features.json'), 'w', encoding='utf-8') as f:
#     json.dump(summoner_champion_features, f, ensure_ascii=False, indent=4)

# cluster_counts = summoner_features_df['cluster'].value_counts()
# print("\n클러스터별 플레이어 수:")
# print(cluster_counts)

# summoner_features_df.to_csv(os.path.join(output_dir, 'hierarchical_cosine_average_cluster_results.csv'))
# cluster_centers.to_csv(os.path.join(output_dir, 'hierarchical_cosine_average_cluster_centers.csv'))

# ### [STEP 7] t-SNE 2D 시각화 (클러스터 결과 확인)
# tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
# summoner_features_2d = tsne.fit_transform(summoner_features_scaled)

# plt.figure(figsize=(10, 7))
# plt.scatter(summoner_features_2d[:, 0], summoner_features_2d[:, 1], c=cluster_labels, cmap='tab10')
# plt.title('t-SNE 2D (Hierarchical - Average Linkage + Cosine)')
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.colorbar(label='cluster')
# plt.savefig(os.path.join(output_dir, 'tsne_2d_cosine_average.png'))
# plt.show()

# ### [STEP 8] t-SNE 3D 시각화
# from mpl_toolkits.mplot3d import Axes3D

# tsne_3d = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
# summoner_features_3d = tsne_3d.fit_transform(summoner_features_scaled)

# fig = plt.figure(figsize=(10,7))
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(summoner_features_3d[:,0], summoner_features_3d[:,1], summoner_features_3d[:,2], c=cluster_labels, cmap='tab10')
# plt.title('t-SNE 3D (Hierarchical - Average Linkage + Cosine)')
# ax.set_xlabel('Dim1')
# ax.set_ylabel('Dim2')
# ax.set_zlabel('Dim3')
# plt.colorbar(sc, label='cluster')
# plt.savefig(os.path.join(output_dir, 'tsne_3d_cosine_average.png'))
# plt.show()

# ### [STEP 9] PCA 시각화 (2D)
# pca_2d = PCA(n_components=2)
# summoner_features_pca_2d = pca_2d.fit_transform(summoner_features_scaled)

# plt.figure(figsize=(10,7))
# plt.scatter(summoner_features_pca_2d[:,0], summoner_features_pca_2d[:,1], c=cluster_labels, cmap='tab10')
# plt.title('PCA 2D (Hierarchical - Average Linkage + Cosine)')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.colorbar(label='cluster')
# plt.savefig(os.path.join(output_dir, 'pca_2d_cosine_average.png'))
# plt.show()

# ### [STEP 10] PCA 3D 시각화
# pca_3d = PCA(n_components=3)
# summoner_features_pca_3d = pca_3d.fit_transform(summoner_features_scaled)

# fig = plt.figure(figsize=(10,7))
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(summoner_features_pca_3d[:,0], summoner_features_pca_3d[:,1], summoner_features_pca_3d[:,2], c=cluster_labels, cmap='tab10')
# plt.title('PCA 3D (Hierarchical - Average Linkage + Cosine)')
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
# plt.colorbar(sc, label='cluster')
# plt.savefig(os.path.join(output_dir, 'pca_3d_cosine_average.png'))
# plt.show()











########################################################################################################################

# import json
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# import os

# from scipy.spatial.distance import pdist
# from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# ### [STEP 1] 챔피언 특성 및 플레이어별 가중치 데이터 로드
# with open('SSW/updated_processed_champion_features.json', 'r', encoding='utf-8') as f:
#     champion_features = json.load(f)

# with open('SSW/scale_summoner_weights.json', 'r', encoding='utf-8') as f:
#     summoner_weights = json.load(f)

# # 결과 디렉토리 생성
# output_dir = 'SSW/New_Hierarchical_Cosine_Avg'
# os.makedirs(output_dir, exist_ok=True)

# ### [STEP 2] 챔피언 특성 DataFrame 생성
# champion_features_df = pd.DataFrame.from_dict(champion_features, orient='index')
# champion_features_df.index.name = 'championId'
# champion_features_df.index = champion_features_df.index.astype(str)

# ### [STEP 3] 소환사별 특성 벡터 및 챔피언별 가중치 특성 계산
# summoner_features = {}
# summoner_champion_features = {}

# for puuid, weights_info in summoner_weights.items():
#     champion_ids = weights_info['championIds']
#     weights = np.array(weights_info['weights'])

#     try:
#         champ_features = champion_features_df.loc[champion_ids]
#     except KeyError as e:
#         print(f"챔피언 ID {e} 데이터 없음")
#         continue

#     weighted_features = champ_features.mul(weights, axis=0)
#     summoner_feature_vector = weighted_features.sum()
#     summoner_features[puuid] = summoner_feature_vector

#     summoner_champion_features[puuid] = {}
#     for idx, cid in enumerate(champion_ids):
#         w = weights[idx]
#         features = champ_features.loc[cid]
#         weighted_feature = features * w
#         summoner_champion_features[puuid][cid] = weighted_feature.to_dict()

# ### [STEP 4] 전처리 (스케일링)
# summoner_features_df = pd.DataFrame.from_dict(summoner_features, orient='index')
# summoner_features_df = summoner_features_df.dropna()

# scaler = StandardScaler()
# summoner_features_scaled = scaler.fit_transform(summoner_features_df)

# ### [STEP 5] 거리행렬 계산 및 계층적 군집화 (Average Linkage + Cosine)
# # pdist로 코사인 거리 계산
# dist_matrix = pdist(summoner_features_scaled, metric='cosine')

# # Average linkage로 계층 군집
# Z = linkage(dist_matrix, method='average')

# ### [STEP 6] 덴드로그램 시각화
# plt.figure(figsize=(10, 7))
# # 덴드로그램 전체를 그리고 싶다면 truncate_mode='lastp'는 생략 가능
# dendrogram(Z, truncate_mode='lastp', p=30)
# plt.title('Dendrogram (Average Linkage + Cosine)')
# plt.xlabel('Player')
# plt.ylabel('Distance')
# plt.savefig(os.path.join(output_dir, 'dendrogram_cosine_average.png'))
# plt.show()

# ### [STEP 7] 거리 임계값(t) 설정을 통한 클러스터 결정
# # t 값을 적절히 조정하며 클러스터 개수 변화
# t = 0.1  # 예시로 0.5 (cosine 거리기 때문에 0~2 범위 내 값)
# cluster_labels = fcluster(Z, t=t, criterion='distance')
# summoner_features_df['cluster'] = cluster_labels

# ### [STEP 7.1] 실루엣 계수 계산
# # 실루엣 계수 계산 시 metric='cosine'과 동일하게
# sil_score = silhouette_score(summoner_features_scaled, cluster_labels, metric='cosine')
# print(f"Silhouette Score (Average+Cosine): {sil_score:.4f}")
# with open(os.path.join(output_dir, 'silhouette_score.txt'), 'w', encoding='utf-8') as f:
#     f.write(f"Silhouette Score (Average+Cosine): {sil_score:.4f}\n")

# ### [STEP 8] 결과 저장 (JSON, CSV)
# summoner_cluster_results = summoner_features_df['cluster'].to_dict()
# with open(os.path.join(output_dir, 'hierarchical_cosine_average_cluster_results.json'), 'w', encoding='utf-8') as f:
#     json.dump(summoner_cluster_results, f, ensure_ascii=False, indent=4)

# cluster_centers = summoner_features_df.groupby('cluster').mean()
# cluster_centers_dict = cluster_centers.to_dict(orient='index')
# with open(os.path.join(output_dir, 'hierarchical_cosine_average_cluster_centers.json'), 'w', encoding='utf-8') as f:
#     json.dump(cluster_centers_dict, f, ensure_ascii=False, indent=4)

# with open(os.path.join(output_dir, 'hierarchical_cosine_average_summoner_champion_weighted_features.json'), 'w', encoding='utf-8') as f:
#     json.dump(summoner_champion_features, f, ensure_ascii=False, indent=4)

# cluster_counts = summoner_features_df['cluster'].value_counts()
# print("\n클러스터별 플레이어 수:")
# print(cluster_counts)

# summoner_features_df.to_csv(os.path.join(output_dir, 'hierarchical_cosine_average_cluster_results.csv'))
# cluster_centers.to_csv(os.path.join(output_dir, 'hierarchical_cosine_average_cluster_centers.csv'))

# ### [STEP 9] t-SNE 2D 시각화 (클러스터 결과 확인)
# tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
# summoner_features_2d = tsne.fit_transform(summoner_features_scaled)

# plt.figure(figsize=(10, 7))
# plt.scatter(summoner_features_2d[:, 0], summoner_features_2d[:, 1], c=cluster_labels, cmap='tab10')
# plt.title('t-SNE 2D (Hierarchical - Average Linkage + Cosine + Distance Threshold)')
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.colorbar(label='cluster')
# plt.savefig(os.path.join(output_dir, 'tsne_2d_cosine_average.png'))
# plt.show()

# ### [STEP 10] t-SNE 3D 시각화
# from mpl_toolkits.mplot3d import Axes3D

# tsne_3d = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
# summoner_features_3d = tsne_3d.fit_transform(summoner_features_scaled)

# fig = plt.figure(figsize=(10,7))
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(summoner_features_3d[:,0], summoner_features_3d[:,1], summoner_features_3d[:,2], c=cluster_labels, cmap='tab10')
# plt.title('t-SNE 3D (Hierarchical - Average Linkage + Cosine + Distance Threshold)')
# ax.set_xlabel('Dim1')
# ax.set_ylabel('Dim2')
# ax.set_zlabel('Dim3')
# plt.colorbar(sc, label='cluster')
# plt.savefig(os.path.join(output_dir, 'tsne_3d_cosine_average.png'))
# plt.show()

# ### [STEP 11] PCA 2D 시각화
# pca_2d = PCA(n_components=2)
# summoner_features_pca_2d = pca_2d.fit_transform(summoner_features_scaled)

# plt.figure(figsize=(10,7))
# plt.scatter(summoner_features_pca_2d[:,0], summoner_features_pca_2d[:,1], c=cluster_labels, cmap='tab10')
# plt.title('PCA 2D (Hierarchical - Average+Cosine+Distance Threshold)')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.colorbar(label='cluster')
# plt.savefig(os.path.join(output_dir, 'pca_2d_cosine_average.png'))
# plt.show()

# ### [STEP 12] PCA 3D 시각화
# pca_3d = PCA(n_components=3)
# summoner_features_pca_3d = pca_3d.fit_transform(summoner_features_scaled)

# fig = plt.figure(figsize=(10,7))
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(summoner_features_pca_3d[:,0], summoner_features_pca_3d[:,1], summoner_features_pca_3d[:,2], c=cluster_labels, cmap='tab10')
# plt.title('PCA 3D (Hierarchical - Average+Cosine+Distance Threshold)')
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
# plt.colorbar(sc, label='cluster')
# plt.savefig(os.path.join(output_dir, 'pca_3d_cosine_average.png'))
# plt.show()




#############################################################################################################

import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

### [STEP 1] 데이터 로드
with open('SSW/updated_processed_champion_features.json', 'r', encoding='utf-8') as f:
    champion_features = json.load(f)

with open('SSW/scale_summoner_weights.json', 'r', encoding='utf-8') as f:
    summoner_weights = json.load(f)

output_dir = 'SSW/PCA_Then_Hierarchical'
os.makedirs(output_dir, exist_ok=True)

### [STEP 2] 챔피언 특성 DataFrame
champion_features_df = pd.DataFrame.from_dict(champion_features, orient='index')
champion_features_df.index.name = 'championId'
champion_features_df.index = champion_features_df.index.astype(str)

### [STEP 3] 소환사별 특성 벡터
summoner_features = {}
summoner_champion_features = {}

for puuid, weights_info in summoner_weights.items():
    champion_ids = weights_info['championIds']
    weights = np.array(weights_info['weights'])

    try:
        champ_features = champion_features_df.loc[champion_ids]
    except KeyError:
        continue

    weighted_features = champ_features.mul(weights, axis=0)
    summoner_feature_vector = weighted_features.sum()
    summoner_features[puuid] = summoner_feature_vector

    summoner_champion_features[puuid] = {}
    for idx, cid in enumerate(champion_ids):
        w = weights[idx]
        features = champ_features.loc[cid]
        weighted_feature = features * w
        summoner_champion_features[puuid][cid] = weighted_feature.to_dict()

### [STEP 4] 전처리 (스케일링)
summoner_features_df = pd.DataFrame.from_dict(summoner_features, orient='index').dropna()
scaler = StandardScaler()
summoner_features_scaled = scaler.fit_transform(summoner_features_df)

### [STEP 5] PCA로 차원 축소
pca_components = 2  # 예: 26 → 10차원 축소
pca = PCA(n_components=pca_components)
X_pca = pca.fit_transform(summoner_features_scaled)

### [STEP 6] 계층적 군집화 - Average + Cosine 거리
dist_matrix = pdist(X_pca, metric='cosine')
Z = linkage(dist_matrix, method='average')

### [STEP 7] 덴드로그램 시각화
plt.figure(figsize=(10,7))
dendrogram(Z, truncate_mode='lastp', p=30)
plt.title(f'Dendrogram (Average+Cosine) with PCA({pca_components} comps)')
plt.xlabel('Player')
plt.ylabel('Distance')
plt.savefig(os.path.join(output_dir, 'dendrogram_cosine_average_pca.png'))
plt.show()

### [STEP 8] 거리 임계값 설정으로 클러스터 결정
t = 0.24  # 예시 값, 덴드로그램 참고하여 조정
cluster_labels = fcluster(Z, t=t, criterion='distance')
summoner_features_df['cluster'] = cluster_labels

### [STEP 9] 실루엣 계수 계산
sil_score = silhouette_score(X_pca, cluster_labels, metric='cosine')
print(f"Silhouette Score (Average+Cosine+PCA): {sil_score:.4f}")
with open(os.path.join(output_dir, 'silhouette_score.txt'), 'w', encoding='utf-8') as f:
    f.write(f"Silhouette Score (Average+Cosine+PCA): {sil_score:.4f}\n")

### [STEP 10] 결과 저장
summoner_cluster_results = summoner_features_df['cluster'].to_dict()
with open(os.path.join(output_dir, 'hierarchical_cosine_average_pca_results.json'), 'w', encoding='utf-8') as f:
    json.dump(summoner_cluster_results, f, ensure_ascii=False, indent=4)

cluster_centers = summoner_features_df.groupby('cluster').mean()
cluster_centers_dict = cluster_centers.to_dict(orient='index')
with open(os.path.join(output_dir, 'hierarchical_cosine_average_pca_centers.json'), 'w', encoding='utf-8') as f:
    json.dump(cluster_centers_dict, f, ensure_ascii=False, indent=4)

with open(os.path.join(output_dir, 'hierarchical_cosine_average_pca_weighted_features.json'), 'w', encoding='utf-8') as f:
    json.dump(summoner_champion_features, f, ensure_ascii=False, indent=4)

cluster_counts = summoner_features_df['cluster'].value_counts()
print("\n클러스터별 플레이어 수:")
print(cluster_counts)

summoner_features_df.to_csv(os.path.join(output_dir, 'hierarchical_cosine_average_pca_results.csv'))
cluster_centers.to_csv(os.path.join(output_dir, 'hierarchical_cosine_average_pca_centers.csv'))

### [STEP 11] t-SNE 2D 시각화 (PCA 후 데이터 기준)
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
summoner_features_2d = tsne.fit_transform(X_pca)

plt.figure(figsize=(10, 7))
plt.scatter(summoner_features_2d[:, 0], summoner_features_2d[:, 1], c=cluster_labels, cmap='tab10')
plt.title('t-SNE 2D (Hierarchical - Avg+Cosine+PCA+Distance Threshold)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.colorbar(label='cluster')
plt.savefig(os.path.join(output_dir, 'tsne_2d_cosine_average_pca.png'))
plt.show()

### [STEP 12] PCA 2D 시각화 (이미 PCA 적용)
# 이미 X_pca가 PCA 결과이므로 여기서 추가로 2D 시각화는 X_pca의 첫 두 주성분만 plotting 가능
plt.figure(figsize=(10,7))
plt.scatter(X_pca[:,0], X_pca[:,1], c=cluster_labels, cmap='tab10')
plt.title('PCA 2D (After Dimensionality Reduction)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(label='cluster')
plt.savefig(os.path.join(output_dir, 'pca_2d_after_pca.png'))
plt.show()

