import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
from sklearn.manifold import TSNE

# 챔피언 특성 데이터 로드
with open('updated_processed_champion_features.json', 'r', encoding='utf-8') as f:
    champion_features = json.load(f)

# 플레이어별 가중치 데이터 로드
with open('scale_summoner_weights.json', 'r', encoding='utf-8') as f:
    summoner_weights = json.load(f)

# 챔피언 특성 데이터프레임 생성
champion_features_df = pd.DataFrame.from_dict(champion_features, orient='index')
champion_features_df.index.name = 'championId'
champion_features_df.index = champion_features_df.index.astype(str)

# 소환사별 특성 벡터 생성
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

# 소환사 특성 데이터프레임 생성
summoner_features_df = pd.DataFrame.from_dict(summoner_features, orient='index')

# 결측치 처리 (필요한 경우)
summoner_features_df = summoner_features_df.dropna()

# 특성 스케일링 (표준화)
scaler = StandardScaler()
summoner_features_scaled = scaler.fit_transform(summoner_features_df)

# 연결 행렬 계산 (유클리드 거리와 워드 연결법 사용)
linked = linkage(summoner_features_scaled, method='ward')

# 덴드로그램 그리기
plt.figure(figsize=(10, 7))
# 복잡성을 줄이기 위해 상위 30개 클러스터만 표시
dendrogram(linked, truncate_mode='lastp', p=30)
plt.title('Dendrogram')
plt.xlabel('Player')
plt.ylabel('Linkage')
plt.show()

# 덴드로그램 그릴 시 트렁케이션 기법 사용하여 표시 가능
# plt.figure(figsize=(10, 7))
# dendrogram(linked, truncate_mode='level', p=5)
# plt.title('Dendrogram (Truncated)')
# plt.xlabel('Sample Index')
# plt.ylabel('Distance')
# plt.show()


# 덴드로그램을 참고하여 거리 임계값 설정 (t = 40)
cluster_labels = fcluster(linked, t=40, criterion='distance')

# 결과를 데이터프레임에 추가
summoner_features_df['cluster'] = cluster_labels

# 데이터프레임에서 플레이어별 클러스터 정보를 딕셔너리로 변환
summoner_cluster_results = summoner_features_df['cluster'].to_dict()

# JSON 파일로 저장
with open('hierarchical_summoner_cluster_results.json', 'w', encoding='utf-8') as f:
    json.dump(summoner_cluster_results, f, ensure_ascii=False, indent=4)

# 클러스터별 평균 특성을 JSON 파일로 저장
# 클러스터별 평균 특성 계산
cluster_centers = summoner_features_df.groupby('cluster').mean()

# 데이터프레임을 딕셔너리로 변환
cluster_centers_dict = cluster_centers.to_dict(orient='index')

# JSON 파일로 저장
with open('hierarchical_cluster_centers.json', 'w', encoding='utf-8') as f:
    json.dump(cluster_centers_dict, f, ensure_ascii=False, indent=4)


# 클러스터별 플레이어 수 확인
cluster_counts = summoner_features_df['cluster'].value_counts()
print("\n클러스터별 플레이어 수:")
print(cluster_counts)

# 결과 저장
summoner_features_df.to_csv('hierarchical_summoner_cluster_results.csv')
cluster_centers.to_csv('hierarchical_cluster_centers.csv')


# t-SNE를 사용하여 데이터 시각화
tsne = TSNE(n_components=2, random_state=42)
summoner_features_2d = tsne.fit_transform(summoner_features_scaled)

# 시각화
plt.figure(figsize=(10, 7))
plt.scatter(summoner_features_2d[:, 0], summoner_features_2d[:, 1], c=cluster_labels, cmap='tab10')
plt.title('t-SNE')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.colorbar(label='cluster')
plt.show()




###### 군집화 시에 클러스터 수를 지정할 수 있음 (서비스 측면에서 고려)
# from sklearn.cluster import AgglomerativeClustering

# # 클러스터 수를 5개로 지정하여 군집화 수행
# n_clusters = 5
# cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
# cluster_labels = cluster.fit_predict(summoner_features_scaled)