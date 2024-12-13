import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import pickle

# 1. 데이터 로드
with open('SSW/updated_processed_champion_features.json', 'r', encoding='utf-8') as f:
    champion_features = json.load(f)

with open('SSW/scale_summoner_weights.json', 'r', encoding='utf-8') as f:
    summoner_weights = json.load(f)

output_dir = 'Clustering/Results_Euclidean'
os.makedirs(output_dir, exist_ok=True)

# 2. 챔피언 특성 DataFrame
champion_features_df = pd.DataFrame.from_dict(champion_features, orient='index')
champion_features_df.index.name = 'championId'
champion_features_df.index = champion_features_df.index.astype(str)

# 3. 소환사별 특성 벡터
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
    summoner_feature_vector = weighted_features.sum()  # 가중합
    summoner_features[puuid] = summoner_feature_vector

    # 챔피언별 가중치 적용 특성 저장
    summoner_champion_features[puuid] = {}
    for idx, cid in enumerate(champion_ids):
        w = weights[idx]
        features = champ_features.loc[cid]
        weighted_feature = features * w
        summoner_champion_features[puuid][cid] = weighted_feature.to_dict()

# 4. 전처리 (스케일링)
summoner_features_df = pd.DataFrame.from_dict(summoner_features, orient='index').dropna()

# columns_order 저장(추후 서비스 때 순서 맞추기용)
columns_order = summoner_features_df.columns.tolist()
with open(os.path.join(output_dir, 'columns_order.json'), 'w', encoding='utf-8') as f:
    json.dump(columns_order, f, ensure_ascii=False, indent=4)

scaler = StandardScaler()
summoner_features_scaled = scaler.fit_transform(summoner_features_df)

# scaler 저장(pickle)
with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

# 5. PCA로 차원 축소
pca_components = 2
pca = PCA(n_components=pca_components)
X_pca = pca.fit_transform(summoner_features_scaled)

# pca 저장
with open(os.path.join(output_dir, 'pca.pkl'), 'wb') as f:
    pickle.dump(pca, f)

# 6. 계층적 군집화 - Average + Euclidean 거리
dist_matrix = pdist(X_pca, metric='euclidean')
Z = linkage(dist_matrix, method='average')

# 덴드로그램
plt.figure(figsize=(10, 7))
dendrogram(Z, truncate_mode='lastp', p=30)
plt.title(f'Dendrogram (Average+Euclidean) with PCA({pca_components} comps)')
plt.xlabel('Player')
plt.ylabel('Distance')
plt.savefig(os.path.join(output_dir, 'dendrogram_euclidean_average_pca.png'))
plt.show()

# 8. 거리 임계값 설정
t = 2.8  # 임계값, 덴드로그램을 보고 조정
cluster_labels = fcluster(Z, t=t, criterion='distance')
summoner_features_df['cluster'] = cluster_labels

# 클러스터 레이블 리스트 정렬
cluster_labels_list = sorted(set(cluster_labels), key=lambda x: int(x))

# 9. 실루엣 계수
sil_score = silhouette_score(X_pca, cluster_labels, metric='euclidean')
print(f"Silhouette Score (Average+Euclidean+PCA): {sil_score:.4f}")
with open(os.path.join(output_dir, 'silhouette_score.txt'), 'w', encoding='utf-8') as f:
    f.write(f"Silhouette Score (Average+Euclidean+PCA): {sil_score:.4f}\n")

# 10. 결과 저장
summoner_cluster_results = summoner_features_df['cluster'].to_dict()
with open(os.path.join(output_dir, 'hierarchical_euclidean_average_pca_results.json'), 'w', encoding='utf-8') as f:
    json.dump(summoner_cluster_results, f, ensure_ascii=False, indent=4)

# 클러스터 센터 계산 (PCA 공간에서 평균)
# 클러스터 센터 계산 (PCA 공간 및 원본 공간)
# 클러스터 센터 계산 (PCA 공간 및 원본 공간)
cluster_centers_pca = []
cluster_centers_original = []  # 원본 공간에서 평균 저장

for clust in cluster_labels_list:
    # PCA 공간 클러스터 멤버
    cluster_members_pca = X_pca[cluster_labels == clust]
    cluster_center_pca = cluster_members_pca.mean(axis=0)
    cluster_centers_pca.append(cluster_center_pca)

    # 원본 공간 클러스터 멤버
    cluster_members_original = summoner_features_df[summoner_features_df['cluster'] == clust]
    cluster_center_original = cluster_members_original[columns_order].mean(axis=0).values  # 열 순서 맞추기
    cluster_centers_original.append(cluster_center_original)

cluster_centers_pca = np.array(cluster_centers_pca)  # (n_clusters, n_pca_components)
cluster_centers_original = np.array(cluster_centers_original)  # (n_clusters, original_feature_count)

# 클러스터 센터 저장 (PCA 공간)
with open(os.path.join(output_dir, 'hierarchical_euclidean_average_pca_centers_pca_space.json'), 'w', encoding='utf-8') as f:
    json.dump(
        {int(clust): center.tolist() for clust, center in zip(cluster_labels_list, cluster_centers_pca)},
        f,
        ensure_ascii=False,
        indent=4
    )

# 클러스터 센터 저장 (원본 스케일 공간)
with open(os.path.join(output_dir, 'hierarchical_euclidean_average_centers_original_space.json'), 'w', encoding='utf-8') as f:
    json.dump(
        {int(clust): center.tolist() for clust, center in zip(cluster_labels_list, cluster_centers_original)},
        f,
        ensure_ascii=False,
        indent=4
    )

# CSV 저장
pd.DataFrame(cluster_centers_pca, index=cluster_labels_list, columns=[f'PCA_{i+1}' for i in range(pca_components)]).to_csv(os.path.join(output_dir, 'hierarchical_euclidean_average_pca_centers_pca_space.csv'))
pd.DataFrame(cluster_centers_original, index=cluster_labels_list, columns=columns_order).to_csv(os.path.join(output_dir, 'hierarchical_euclidean_average_centers_original_space.csv'))

# PCA 2D 시각화
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10')
plt.title('PCA 2D (After Dimensionality Reduction)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(label='cluster')
plt.savefig(os.path.join(output_dir, 'pca_2d_after_pca.png'))
plt.show()
