import json
import numpy as np
import pickle
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns

# 파일 로드
with open('SSW/updated_processed_champion_features.json', 'r', encoding='utf-8') as f:
    champion_features = json.load(f)

with open('Clustering\Results_Euclidean\columns_order.json', 'r', encoding='utf-8') as f:
    columns_order = json.load(f)

with open('Clustering\Results_Euclidean\scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('Clustering\Results_Euclidean\pca.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('Clustering\Results_Euclidean\hierarchical_euclidean_average_pca_centers_pca_space.json', 'r', encoding='utf-8') as f:
    cluster_centers_pca_dict = json.load(f)

# 클러스터 레이블 리스트 정렬 및 int 변환
cluster_labels_list = sorted(cluster_centers_pca_dict.keys(), key=lambda x: int(x))
cluster_centers_pca = np.array([cluster_centers_pca_dict[str(clust)] for clust in cluster_labels_list])  # (n_clusters, n_pca_components)

def assign_cluster_to_player(player_data):
    """
    player_data 형태:
    {
      "name": "한유민",
      "nickname": "자크빼면시체",
      "tag": "KR1",
      "puuid": "...",
      "topChampions": [
         {"championId": 154, "championPoints": 357776},
         {"championId": 81, "championPoints": 117099},
         {"championId": 80, "championPoints": 76429}
      ]
    }

    이 정보를 이용해 summoner_feature_vector를 만들고
    scaler, pca 적용 후 cluster_centers_pca와의 거리 비교로 클러스터 할당
    """

    top_champs = player_data["topChampions"]
    weighted_sum = np.zeros(len(columns_order))
    total_points = 0  # 총 챔피언 포인트를 추적

    # 챔피언별 데이터 수집
    champ_data = []
    for ch in top_champs:
        cid = str(ch["championId"])
        cpoints = ch["championPoints"]
        total_points += cpoints
        cfeat = champion_features.get(cid, {})
        champ_vec = np.array([cfeat.get(col, 0) for col in columns_order])
        weighted_sum += champ_vec * cpoints
        champ_data.append({'Champion ID': cid, 'Points': cpoints})
        print(f"Champion ID: {cid}, Points: {cpoints}")

    if total_points == 0:
        raise ValueError("Total champion points are zero. Cannot normalize summoner vector.")

    # 챔피언 포인트 비율 계산
    for data in champ_data:
        data['Percentage'] = (data['Points'] / total_points) * 100

    # 챔피언 기여도 출력
    champ_df = pd.DataFrame(champ_data)
    print("\n=== Champion Contributions ===")
    print(champ_df.to_string(index=False))

    # 총 챔피언 포인트로 나누어 평균 특성 벡터 생성
    summoner_vector = weighted_sum / total_points
    print("\nSummoner Vector (평균화된 특성 벡터):", summoner_vector)

    # Summoner Vector와 Columns Order 확인
    print("\n=== Summoner Vector with Columns ===")
    for col, val in zip(columns_order, summoner_vector):
        print(f"{col}: {val}")

    # 챔피언 포인트 비율 시각화
    plt.figure(figsize=(10,6))
    champions = champ_df['Champion ID'].astype(int).astype(str)  # Champion ID를 정수형으로 변환 후 문자열로
    percentages = champ_df['Percentage']
    sns.barplot(x=champions, y=percentages, palette='viridis')
    plt.xlabel('Champion ID')
    plt.ylabel('Champion Points Percentage (%)')
    plt.title('Champion Points Distribution')
    plt.show()

    # Summoner Vector DataFrame 출력
    summoner_vector_df = pd.DataFrame(summoner_vector.reshape(1, -1), columns=columns_order)
    print("\nSummoner Vector DataFrame:")
    print(summoner_vector_df)

    # 스케일러와 PCA 적용
    summoner_scaled = scaler.transform(summoner_vector_df)
    print("\nSummoner Scaled Vector:", summoner_scaled)

    summoner_pca = pca.transform(summoner_scaled)
    print("\nSummoner PCA Vector:", summoner_pca)

    # 거리 계산 (유클리드 거리)
    dists = cdist(summoner_pca, cluster_centers_pca, metric='euclidean')
    print("\nDistances to Cluster Centers (Euclidean):", dists)

    # 각 클러스터와의 거리 출력
    print("\n=== Euclidean Distances to Each Cluster ===")
    for i, clust in enumerate(cluster_labels_list):
        print(f"클러스터 {clust}와의 유클리드 거리: {dists[0][i]:.4f}")

    # 가장 가까운 클러스터 찾기
    min_idx = np.argmin(dists)
    assigned_cluster = cluster_labels_list[min_idx]
    print(f"\nAssigned Cluster: {assigned_cluster} (Index: {min_idx})")

    # 클러스터 센터와 플레이어 벡터 시각화
    if pca.n_components_ >= 2:
        plt.figure(figsize=(12, 8))
        
        # 클러스터 센터 시각화
        sns.scatterplot(
            x=cluster_centers_pca[:, 0], 
            y=cluster_centers_pca[:, 1], 
            hue=cluster_labels_list, 
            palette='deep', 
            s=100, 
            marker='X',
            legend='full'  # 모든 클러스터 레이블을 표시
        )
        
        # 소환사 벡터 시각화
        sns.scatterplot(
            x=summoner_pca[:, 0], 
            y=summoner_pca[:, 1], 
            color='red', 
            marker='o',
            s=200,
            label='Player'
        )
        
        plt.title("PCA SPACE: Cluster Centers and Player Vector")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.show()

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

try:
    assigned = assign_cluster_to_player(player_data)
    print("\n이 플레이어는 클러스터:", assigned, "에 할당되었습니다.")
    
except ValueError as ve:
    print(f"\nError: {ve}")
except Exception as e:
    print(f"\nUnexpected error: {e}")
