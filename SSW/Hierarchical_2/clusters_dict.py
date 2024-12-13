import json


with open('SSW/PCA_Then_Hierarchical/hierarchical_cosine_average_pca_results.json', 'r', encoding='utf-8') as f:
    puuid_to_cluster = json.load(f)

clusters_dict = {}

for puuid, cluster_id in puuid_to_cluster.items():
    if cluster_id not in clusters_dict:
        clusters_dict[cluster_id] = []
    clusters_dict[cluster_id].append(puuid)

# clusters_dict는 이제 { cluster_id: [puuid들...] } 형태
print(clusters_dict)

with open('SSW/PCA_Then_Hierarchical/clusters_by_id.json', 'w', encoding='utf-8') as f:
    json.dump(clusters_dict, f, ensure_ascii=False, indent=4)