import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns



with open('processed_champion_features.json', 'r', encoding='utf-8') as f:
    champion_features = json.load(f)

# 2. 소환사별 상위 챔피언 ID 로드
with open('top_champions_per_summoner.json', 'r', encoding='utf-8') as f:
    top_champions_per_summoner = json.load(f)

print("챔피언 특성 데이터 예시:")
for champ_id, features in list(champion_features.items())[:1]:
    print(f"챔피언 ID: {champ_id}")
    for key, value in features.items():
        print(f"  {key}: {value}")