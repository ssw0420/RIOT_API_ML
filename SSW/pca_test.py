import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# 1. 챔피언 특성 데이터 로드
with open('updated_processed_champion_features.json', 'r', encoding='utf-8') as f:
    champion_features = json.load(f)

# 2. 소환사별 상위 챔피언 숙련도 데이터 로드
with open('champion_mastery_top.json', 'r', encoding='utf-8') as f:
    top_champion_mastery_per_summoner = json.load(f)

# 챔피언 특성 데이터프레임 생성
champion_features_df = pd.DataFrame.from_dict(champion_features, orient='index')
champion_features_df.index.name = 'championId'
champion_features_df.index = champion_features_df.index.astype(str)

# 데이터프레임 확인
print("\n챔피언 특성 데이터프레임:")
print(champion_features_df.head())

# 소환사별 특성 데이터를 저장할 딕셔너리
summoner_features = {}

# 전체 숙련도 점수를 수집하여 StandardScaler에 사용
all_mastery_scores_log = []

for puuid, champion_mastery_list in top_champion_mastery_per_summoner.items():
    for champ_info in champion_mastery_list:
        mastery_score = float(champ_info['championPoints'])
        mastery_score_log = np.log1p(mastery_score)
        all_mastery_scores_log.append(mastery_score_log)

# 전체 숙련도 점수 표준화
scaler = StandardScaler()
scaler.fit(np.array(all_mastery_scores_log).reshape(-1, 1))

for puuid, champion_mastery_list in top_champion_mastery_per_summoner.items():
    champion_ids = []
    mastery_scores = []
    
    for champ_info in champion_mastery_list:
        champion_ids.append(str(champ_info['championId']))
        mastery_scores.append(float(champ_info['championPoints']))
    
    # 숙련도 점수 로그 변환
    mastery_scores_log = np.log1p(mastery_scores)
    
    # 숙련도 점수 표준화 (전체 분포 기반)
    mastery_scores_scaled = scaler.transform(np.array(mastery_scores_log).reshape(-1, 1)).flatten()
    
    # 음수를 0으로 변환 (ReLU 함수 적용)
    mastery_scores_positive = np.maximum(mastery_scores_scaled, 0)
    
    # 가중치 계산 (합이 1이 되도록)
    if mastery_scores_positive.sum() == 0:
        # 모든 값이 0인 경우 동일한 가중치 부여
        weights = np.ones_like(mastery_scores_positive) / len(mastery_scores_positive)
    else:
        weights = mastery_scores_positive / mastery_scores_positive.sum()
    
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

# 결측치 처리
summoner_features_df.dropna(inplace=True)

# 수치형 특성 선택 및 정규화
numeric_columns = summoner_features_df.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
summoner_features_scaled = scaler.fit_transform(summoner_features_df[numeric_columns])
summoner_features_scaled_df = pd.DataFrame(summoner_features_scaled, index=summoner_features_df.index, columns=numeric_columns)



# 클러스터 수에 따른 BIC 계산
bic_scores = []
n_components_range = range(1, 20)

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
    gmm.fit(summoner_features_scaled_df)
    bic = gmm.bic(summoner_features_scaled_df)
    bic_scores.append(bic)

# BIC 시각화
plt.figure(figsize=(8, 5))
plt.plot(n_components_range, bic_scores, marker='o')
plt.xlabel('클러스터 수 (n_components)')
plt.ylabel('BIC 점수')
plt.title('최적의 클러스터 수 찾기 (BIC 기준)')
plt.show()


# 가우시안 혼합 모델 적용
n_components = 11  # 클러스터 수 설정 (적절히 변경 가능)
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
gmm.fit(summoner_features_scaled_df)
labels = gmm.predict(summoner_features_scaled_df)

# 클러스터 레이블 저장
summoner_features_df['cluster'] = labels

# 결과 확인
print("\n클러스터 할당 결과:")
print(summoner_features_df[['cluster']].head())

# # 결과를 CSV 파일로 저장
# summoner_features_df.to_csv('gmm_weighted_summoner_clustering_results.csv')

# # 또는 JSON 파일로 저장
# summoner_features_df.to_json('gmm_weighted_summoner_clustering_results.json', orient='index', indent=4)

print("\n클러스터링 결과를 저장했습니다.")

# 클러스터별 평균 특성 계산
cluster_centers = summoner_features_df.groupby('cluster').mean()

print("\n클러스터별 평균 특성:")
print(cluster_centers)

# PCA로 3차원으로 축소
pca = PCA(n_components=3)
principal_components = pca.fit_transform(summoner_features_scaled_df)

# 결과를 데이터프레임으로 생성
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'], index=summoner_features_df.index)
pca_df['cluster'] = summoner_features_df['cluster']

# 3D 시각화
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=pca_df['cluster'], cmap='Set2')
ax.set_title('PCA 3D 시각화')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.legend(*scatter.legend_elements(), title="클러스터")
plt.show()


# 누적 설명된 분산 비율 확인
pca = PCA().fit(summoner_features_scaled_df)
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
plt.xlabel('주성분 개수')
plt.ylabel('누적 설명된 분산 비율')
plt.title('PCA 누적 설명된 분산 비율')
plt.grid(True)
plt.show()
