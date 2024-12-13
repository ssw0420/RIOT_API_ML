from flask import Flask, request, jsonify
import json
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# 런타임에 필요한 파일 로드
with open('../../../SSW/updated_processed_champion_features.json', 'r', encoding='utf-8') as f:
    champion_features = json.load(f)


with open('../../../Clustering/Results/columns_order.json', 'r', encoding='utf-8') as f:
    columns_order = json.load(f)

with open('../../../Clustering/Results/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('../../../Clustering/Results/pca.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('../../../Clustering/Results/hierarchical_cosine_average_pca_centers.json', 'r', encoding='utf-8') as f:
    cluster_centers_dict = json.load(f)

# 클러스터 센터 변환
cluster_labels_list = sorted(cluster_centers_dict.keys(), key=lambda x: int(x))
cluster_centers = []
for clust in cluster_labels_list:
    center_feat = cluster_centers_dict[clust]
    vec = [center_feat[col] for col in columns_order]
    cluster_centers.append(vec)
cluster_centers = np.array(cluster_centers)

# 클러스터 센터를 PCA 공간으로 변환
cc_scaled = scaler.transform(cluster_centers)
cc_pca = pca.transform(cc_scaled)
cluster_centers = cc_pca


# 클러스터 할당 함수
def assign_cluster_to_player(player_data):
    top_champs = player_data["topChampions"]
    weighted_sum = np.zeros(len(columns_order))
    for ch in top_champs:
        cid = str(ch["championId"])
        cpoints = ch["championPoints"]
        cfeat = champion_features.get(cid, {})
        champ_vec = np.array([cfeat.get(col, 0) for col in columns_order])
        weighted_sum += champ_vec * cpoints

    summoner_vector = weighted_sum

    # Scaler와 PCA 적용
    summoner_vector_df = pd.DataFrame(summoner_vector.reshape(1, -1), columns=columns_order)
    summoner_scaled = scaler.transform(summoner_vector_df)
    summoner_pca = pca.transform(summoner_scaled)

    # 클러스터 할당 (유클리드 거리 계산)
    dists = np.linalg.norm(cluster_centers - summoner_pca, axis=1)
    min_idx = np.argmin(dists)
    assigned_cluster = cluster_labels_list[min_idx]
    return assigned_cluster


# Flask 엔드포인트: 클라이언트 데이터 수신 후 클러스터 할당
@app.route('/assign-cluster', methods=['POST'])
def assign_cluster():
    try:
        # 클라이언트로부터 데이터 수신
        player_data = request.json
        print("받은 데이터:", player_data)

        # 클러스터 할당
        assigned_cluster = assign_cluster_to_player(player_data)

        # 결과 반환
        return jsonify({
            "status": "success",
            "assignedCluster": assigned_cluster
        }), 200
    except Exception as e:
        print("오류 발생:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
