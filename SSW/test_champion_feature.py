import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# JSON 파일 로드
with open('updated_processed_champion_features.json', 'r') as file:
    data_json = json.load(file)

# JSON 데이터를 DataFrame으로 변환
data = pd.DataFrame.from_dict(data_json, orient='index')

# 데이터의 첫 몇 행 확인
print(data.head())

# 데이터 구조 확인
print(data.info())

# 결측치 확인
print(data.isnull().sum())

categorical_features = ['Fighter', 'Mage', 'Assassin', 'Marksman', 'Tank', 'Support']
print(data[categorical_features].head())

# 수치형 특성 리스트
numerical_features = [
    'attack', 'defense', 'magic', 'difficulty', 'hp', 'hpperlevel',
    'movespeed', 'armor', 'armorperlevel', 'spellblock', 
    'spellblockperlevel', 'attackrange', 'hpregen', 
    'hpregenperlevel', 'crit', 'critperlevel', 
    'attackdamage', 'attackdamageperlevel', 
    'attackspeedperlevel', 'attackspeed'
]

# 스케일링 수행
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[numerical_features])

# 스케일링된 데이터를 DataFrame으로 변환
data_scaled = pd.DataFrame(data_scaled, columns=numerical_features)

# 범주형 변수 추가
data_scaled = pd.concat([data_scaled, data[categorical_features].reset_index(drop=True)], axis=1)

print(data_scaled.head())

# 상관행렬 계산
corr_matrix = data_scaled.corr()

# 히트맵 시각화
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()


# 수치형 특성의 분포 시각화
import math

num_plots = len(numerical_features)
num_cols = 4
num_rows = math.ceil(num_plots / num_cols)

plt.figure(figsize=(20, 15))
for idx, feature in enumerate(numerical_features):
    plt.subplot(num_rows, num_cols, idx + 1)
    sns.histplot(data_scaled[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()
