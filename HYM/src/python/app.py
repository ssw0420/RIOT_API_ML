import json
import numpy as np
import pickle
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, request, jsonify
import os

# 파일 로드 (작업 디렉토리를 기준으로 경로 조정 필요)
with open('../../../SSW/updated_processed_champion_features.json', 'r', encoding='utf-8') as f:
    champion_features = json.load(f)

with open('../../../Clustering/Results_New/columns_order.json', 'r', encoding='utf-8') as f:
    columns_order = json.load(f)

with open('../../../Clustering/Results_New/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('../../../Clustering/Results_New/pca.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('../../../Clustering/Results_New/hierarchical_cosine_average_pca_centers_pca_space.json', 'r', encoding='utf-8') as f:
    cluster_centers_pca_dict = json.load(f)

# 클러스터 레이블 리스트 정렬 및 int 변환
cluster_labels_list = sorted(cluster_centers_pca_dict.keys(), key=lambda x: int(x))
cluster_centers_pca = np.array([cluster_centers_pca_dict[str(clust)] for clust in cluster_labels_list])  # (n_clusters, n_pca_components)


def assign_cluster_to_player(player_data):
    top_champs = player_data["topChampions"]
    weighted_sum = np.zeros(len(columns_order))
    total_points = 0  # 총 챔피언 포인트 추적

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

    # 평균 특성 벡터 생성
    summoner_vector = weighted_sum / total_points
    print("\nSummoner Vector (평균화된 특성 벡터):", summoner_vector)

    print("\n=== Summoner Vector with Columns ===")
    for col, val in zip(columns_order, summoner_vector):
        print(f"{col}: {val}")

    # 시각화(서버 환경에서는 필요 없을 수 있지만 예제 유지)
    plt.figure(figsize=(10,6))
    champions = champ_df['Champion ID'].astype(int).astype(str)
    percentages = champ_df['Percentage']
    sns.barplot(x=champions, y=percentages, palette='viridis')
    plt.xlabel('Champion ID')
    plt.ylabel('Champion Points Percentage (%)')
    plt.title('Champion Points Distribution')
    # 서버 환경에서는 plt.show()가 블로킹이 될 수 있으니 필요 시 주석처리
    plt.close()

    # Summoner Vector DataFrame
    summoner_vector_df = pd.DataFrame(summoner_vector.reshape(1, -1), columns=columns_order)
    print("\nSummoner Vector DataFrame:")
    print(summoner_vector_df)

    # 스케일러와 PCA 적용
    summoner_scaled = scaler.transform(summoner_vector_df)
    print("\nSummoner Scaled Vector:", summoner_scaled)

    summoner_pca = pca.transform(summoner_scaled)
    print("\nSummoner PCA Vector:", summoner_pca)

    # 거리 계산 (코사인 거리)
    dists = cdist(summoner_pca, cluster_centers_pca, metric='cosine')
    print("\nDistances to Cluster Centers:", dists)

    print("\n=== Cosine Distances to Each Cluster ===")
    for i, clust in enumerate(cluster_labels_list):
        print(f"클러스터 {clust}와의 코사인 거리: {dists[0][i]:.4f}")

    # 가장 가까운 클러스터 선택
    min_idx = np.argmin(dists)
    assigned_cluster = cluster_labels_list[min_idx]
    print(f"\nAssigned Cluster: {assigned_cluster} (Index: {min_idx})")

    # PCA 시각화(서버환경 미사용 시 plt.close())
    if pca.n_components_ >= 2:
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x=cluster_centers_pca[:, 0],
            y=cluster_centers_pca[:, 1],
            hue=cluster_labels_list,
            palette='deep',
            s=100,
            marker='X',
            legend='full'
        )
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
        # 서버환경에서는 plt.show() 대신 close
        plt.close()

    return assigned_cluster


app = Flask(__name__)

@app.route('/send-data', methods=['POST'])
def send_data():
    # JSON 데이터 받기
    data = request.json
    print("받은 데이터:", data)

    try:
        # 클러스터 할당
        assigned_cluster = assign_cluster_to_player(data)
        print("할당된 클러스터:", assigned_cluster)

        return jsonify({
            "status": "success",
            "assignedCluster": assigned_cluster,
            "message": "데이터가 성공적으로 처리되었습니다!"
        }), 200
    except Exception as e:
        print("오류 발생:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    # Flask 서버 실행
    app.run(host="0.0.0.0", port=5000)
