import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from sklearn.metrics import silhouette_score  # 실루엣 계수 계산을 위한 추가

### [STEP 1] 챔피언 특성 및 플레이어별 가중치 데이터 로드
with open('SSW/updated_processed_champion_features.json', 'r', encoding='utf-8') as f:
    champion_features = json.load(f)

with open('SSW/scale_summoner_weights.json', 'r', encoding='utf-8') as f:
    summoner_weights = json.load(f)

# 결과 디렉토리 생성
output_dir = 'SSW/Ward_Hierarchical'
os.makedirs(output_dir, exist_ok=True)

### [STEP 2] 챔피언 특성 DataFrame 생성
champion_features_df = pd.DataFrame.from_dict(champion_features, orient='index')
champion_features_df.index.name = 'championId'
champion_features_df.index = champion_features_df.index.astype(str)

### [STEP 3] 소환사별 특성 벡터 및 챔피언별 가중치 특성 계산
summoner_features = {}
summoner_champion_features = {}

for puuid, weights_info in summoner_weights.items():
    champion_ids = weights_info['championIds']
    weights = np.array(weights_info['weights'])
    
    try:
        champ_features = champion_features_df.loc[champion_ids]
    except KeyError as e:
        print(f"챔피언 ID {e}에 대한 데이터가 없습니다.")
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
summoner_features_df = pd.DataFrame.from_dict(summoner_features, orient='index')
summoner_features_df = summoner_features_df.dropna()

scaler = StandardScaler()
summoner_features_scaled = scaler.fit_transform(summoner_features_df)

### [STEP 5] 계층적 군집화 (Ward 방식)
linked = linkage(summoner_features_scaled, method='ward')

### [STEP 6] 덴드로그램 시각화
plt.figure(figsize=(10, 7))
dendrogram(linked, truncate_mode='lastp', p=30)
plt.title('Dendrogram (Ward)')
plt.xlabel('Player')
plt.ylabel('Linkage')
plt.savefig(os.path.join(output_dir, 'dendrogram_ward.png'))
plt.show()

### [STEP 7] 거리 임계값 설정을 통한 클러스터 결정
t = 5  # 조정 수치
cluster_labels = fcluster(linked, t=t, criterion='distance')
summoner_features_df['cluster'] = cluster_labels

### [STEP 7.1] 실루엣 계수 계산
sil_score = silhouette_score(summoner_features_scaled, cluster_labels)
print(f"Silhouette Score: {sil_score:.4f}")
# 필요하면 결과를 파일에 기록
with open(os.path.join(output_dir, 'silhouette_score.txt'), 'w', encoding='utf-8') as f:
    f.write(f"Silhouette Score: {sil_score:.4f}\n")

### [STEP 8] 결과 저장 (JSON, CSV)
summoner_cluster_results = summoner_features_df['cluster'].to_dict()
with open(os.path.join(output_dir, 'hierarchical_summoner_cluster_results.json'), 'w', encoding='utf-8') as f:
    json.dump(summoner_cluster_results, f, ensure_ascii=False, indent=4)

cluster_centers = summoner_features_df.groupby('cluster').mean()
cluster_centers_dict = cluster_centers.to_dict(orient='index')
with open(os.path.join(output_dir, 'hierarchical_cluster_centers.json'), 'w', encoding='utf-8') as f:
    json.dump(cluster_centers_dict, f, ensure_ascii=False, indent=4)

with open(os.path.join(output_dir, 'hierarchical_summoner_champion_weighted_features.json'), 'w', encoding='utf-8') as f:
    json.dump(summoner_champion_features, f, ensure_ascii=False, indent=4)

cluster_counts = summoner_features_df['cluster'].value_counts()
print("\n클러스터별 플레이어 수:")
print(cluster_counts)

summoner_features_df.to_csv(os.path.join(output_dir, 'hierarchical_summoner_cluster_results.csv'))
cluster_centers.to_csv(os.path.join(output_dir, 'hierarchical_cluster_centers.csv'))

### [STEP 9] t-SNE 2D 시각화
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
summoner_features_2d = tsne.fit_transform(summoner_features_scaled)

plt.figure(figsize=(10, 7))
plt.scatter(summoner_features_2d[:, 0], summoner_features_2d[:, 1], c=cluster_labels, cmap='tab10')
plt.title('t-SNE 2D (Ward)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.colorbar(label='cluster')
plt.savefig(os.path.join(output_dir, 'tsne_2d_ward.png'))
plt.show()

### [STEP 10] t-SNE 3D 시각화
from mpl_toolkits.mplot3d import Axes3D

tsne_3d = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
summoner_features_3d = tsne_3d.fit_transform(summoner_features_scaled)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(summoner_features_3d[:,0], summoner_features_3d[:,1], summoner_features_3d[:,2], c=cluster_labels, cmap='tab10')
plt.title('t-SNE 3D (Ward)')
ax.set_xlabel('Dim1')
ax.set_ylabel('Dim2')
ax.set_zlabel('Dim3')
plt.colorbar(sc, label='cluster')
plt.savefig(os.path.join(output_dir, 'tsne_3d_ward.png'))
plt.show()

### [STEP 11] PCA 2D 시각화
pca_2d = PCA(n_components=2)
summoner_features_pca_2d = pca_2d.fit_transform(summoner_features_scaled)

plt.figure(figsize=(10,7))
plt.scatter(summoner_features_pca_2d[:,0], summoner_features_pca_2d[:,1], c=cluster_labels, cmap='tab10')
plt.title('PCA 2D (Ward)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(label='cluster')
plt.savefig(os.path.join(output_dir, 'pca_2d_ward.png'))
plt.show()

### [STEP 12] PCA 3D 시각화
pca_3d = PCA(n_components=3)
summoner_features_pca_3d = pca_3d.fit_transform(summoner_features_scaled)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(summoner_features_pca_3d[:,0], summoner_features_pca_3d[:,1], summoner_features_pca_3d[:,2], c=cluster_labels, cmap='tab10')
plt.title('PCA 3D (Ward)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.colorbar(sc, label='cluster')
plt.savefig(os.path.join(output_dir, 'pca_3d_ward.png'))
plt.show()

