import json
import numpy as np
import pickle
import pandas as pd

# 런타임에 필요 파일 로드
with open('SSW/updated_processed_champion_features.json', 'r', encoding='utf-8') as f:
    champion_features = json.load(f)

with open('Clustering\Results\columns_order.json', 'r', encoding='utf-8') as f:
    columns_order = json.load(f)

with open('Clustering\Results\scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('Clustering\Results\pca.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('Clustering\Results\hierarchical_cosine_average_pca_centers.json', 'r', encoding='utf-8') as f:
    cluster_centers_dict = json.load(f)

# cluster_centers_dict: { "1": {특성:값, ...}, "2":{...}, ...}
# 이걸 array로 변환 (columns_order 순서)
cluster_labels_list = sorted(cluster_centers_dict.keys(), key=lambda x: int(x))
cluster_centers = []
for clust in cluster_labels_list:
    center_feat = cluster_centers_dict[clust]
    vec = [center_feat[col] for col in columns_order]
    cluster_centers.append(vec)
cluster_centers = np.array(cluster_centers)  # (n_clusters, n_features)

# PCA 적용한 상태에서 센터를 구했다고 가정 안했는데 어쩌지?
# 위 초기 코드에서는 cluster_centers를 summoner_features_df.groupby('cluster').mean() 했는데
# 이 mean은 PCA 적용 전 데이터인지 후 데이터인지 확인 필요.
# 현재 mean은 PCA 전 원본 특징공간에서 계산됨. 그러면 동일하게 새 점도 원본 특징공간에서 dist 비교해야 함.
# 하지만 지금 pca까지 적용했으니, consistency를 위해 cluster_centers도 PCA 공간으로 변환해야 한다.
# cluster_centers 현재 원본 스페이스 mean이므로 transform 필요
# cluster_centers → scaler.transform → pca.transform 하면 PCA 공간으로 갈 수 있음

# cluster_centers array -> reshape & transform
cc_scaled = scaler.transform(cluster_centers)
cc_pca = pca.transform(cc_scaled)
cluster_centers = cc_pca  # 이제 PCA 공간에서의 센터 좌표

def assign_cluster_to_player(player_data):
    # """
    # player_data 형태:
    # {
    #   "name": "한유민",
    #   "nickname": "자크빼면시체",
    #   "tag": "KR1",
    #   "puuid": "...",
    #   "topChampions": [
    #      {"championId": 154, "championPoints": 357776},
    #      {"championId": 81, "championPoints": 117099},
    #      {"championId": 80, "championPoints": 76429}
    #   ]
    # }

    # 이 정보를 이용해 summoner_feature_vector를 만들고
    # scaler, pca 적용 후 cluster_centers와의 거리 비교로 클러스터 할당
    # """

    top_champs = player_data["topChampions"]
    weighted_sum = np.zeros(len(columns_order))
    for ch in top_champs:
        cid = str(ch["championId"])
        cpoints = ch["championPoints"]
        cfeat = champion_features.get(cid, {})
        champ_vec = np.array([cfeat.get(col,0) for col in columns_order])
        weighted_sum += champ_vec * cpoints

    summoner_vector = weighted_sum  # 이전 로직에 맞춰 sum 사용

    # scaler, pca 적용
    summoner_vector_df = pd.DataFrame(summoner_vector.reshape(1,-1), columns=columns_order)
    summoner_scaled = scaler.transform(summoner_vector_df)
    summoner_pca = pca.transform(summoner_scaled)

    # 거리 계산 (유클리드)
    dists = np.linalg.norm(cluster_centers - summoner_pca, axis=1)
    min_idx = np.argmin(dists)
    assigned_cluster = cluster_labels_list[min_idx]
    return assigned_cluster

# 예시 실행
player_data = {
 "name": "한유민",
 "nickname": "자크빼면시체",
 "tag": "KR1",
 "puuid": "kQxfpLmp3R4QfIVqE5Yh88ZV48h_zHGBfUR_ElJF7JR_MY_5jWJeYXn1yJkhN4_3-NUbSAo3MKNKA",
 "topChampions": [
    {"championId": 154, "championPoints": 357776},
    {"championId": 81, "championPoints": 117099},
    {"championId": 80, "championPoints": 76429}
 ]
}

assigned = assign_cluster_to_player(player_data)
print("This player is assigned to cluster:", assigned)
