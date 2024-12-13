import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import pickle

# 1. 데이터 로드
with open('SSW/updated_processed_champion_features.json', 'r', encoding='utf-8') as f:
    champion_features = json.load(f)

with open('SSW\scale_summoner_weights.json', 'r', encoding='utf-8') as f:
    summoner_weights = json.load(f)

output_dir = 'Clustering/Results'
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

# 6. 계층적 군집화 - Average + Cosine 거리
dist_matrix = pdist(X_pca, metric='cosine')
Z = linkage(dist_matrix, method='average')

# 덴드로그램
plt.figure(figsize=(10,7))
dendrogram(Z, truncate_mode='lastp', p=30)
plt.title(f'Dendrogram (Average+Cosine) with PCA({pca_components} comps)')
plt.xlabel('Player')
plt.ylabel('Distance')
plt.savefig(os.path.join(output_dir, 'dendrogram_cosine_average_pca.png'))
plt.show()

# 8. 거리 임계값 설정
t = 0.24
cluster_labels = fcluster(Z, t=t, criterion='distance')
summoner_features_df['cluster'] = cluster_labels

# 9. 실루엣 계수
sil_score = silhouette_score(X_pca, cluster_labels, metric='cosine')
print(f"Silhouette Score (Average+Cosine+PCA): {sil_score:.4f}")
with open(os.path.join(output_dir, 'silhouette_score.txt'), 'w', encoding='utf-8') as f:
    f.write(f"Silhouette Score (Average+Cosine+PCA): {sil_score:.4f}\n")

# 10. 결과 저장
summoner_cluster_results = summoner_features_df['cluster'].to_dict()
with open(os.path.join(output_dir, 'hierarchical_cosine_average_pca_results.json'), 'w', encoding='utf-8') as f:
    json.dump(summoner_cluster_results, f, ensure_ascii=False, indent=4)

# 클러스터 센터 계산 (PCA 공간에서 평균)
cluster_centers = summoner_features_df.groupby('cluster').mean()
cluster_centers_dict = cluster_centers.to_dict(orient='index')
with open(os.path.join(output_dir, 'hierarchical_cosine_average_pca_centers.json'), 'w', encoding='utf-8') as f:
    json.dump(cluster_centers_dict, f, ensure_ascii=False, indent=4)

with open(os.path.join(output_dir, 'hierarchical_cosine_average_pca_weighted_features.json',), 'w', encoding='utf-8') as f:
    json.dump(summoner_champion_features, f, ensure_ascii=False, indent=4)

cluster_counts = summoner_features_df['cluster'].value_counts()
print("\n클러스터별 플레이어 수:")
print(cluster_counts)

summoner_features_df.to_csv(os.path.join(output_dir, 'hierarchical_cosine_average_pca_results.csv'))
cluster_centers.to_csv(os.path.join(output_dir, 'hierarchical_cosine_average_pca_centers.csv'))

# 시각화는 생략 가능하지만 유지
# ... (t-SNE 등 시각화) ...

# 이로써 초기 군집화 과정 완료, scaler.pkl, pca.pkl, columns_order.json, cluster_centers.json 등 생성됨.


