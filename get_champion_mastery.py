import requests
import json
from urllib import parse
from dotenv import load_dotenv
import os
import time

load_dotenv()

api_key = os.getenv('RIOT_API_KEY')

if not api_key:
    raise ValueError("API 키가 설정되지 않았습니다. 환경 변수를 확인하세요.")

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "https://developer.riotgames.com",
    "X-Riot-Token": api_key
}

with open('puuid_list.json', 'r', encoding='utf-8') as f:
    puuid_data = json.load(f)


summoner_data_dict = {}


requests_made = 0
start_time_1s = time.time()
start_time_2m = time.time()

# 총 요청할 ID 수
total_ids = len(puuid_data)

for i, puuid in enumerate(puuid_data):
    lol_champion_mastery_top_url = f"https://kr.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/top?count=3"

    try:
        # API 요청
        response = requests.get(lol_champion_mastery_top_url, headers=REQUEST_HEADERS)
        response.raise_for_status()  # HTTP 에러 발생 시 예외 발생

        # 응답 JSON 파싱
        data = response.json()

        # 데이터 리스트에 추가
        summoner_data_dict[puuid] = data

        # 요청 횟수 증가
        requests_made += 1

        # 20번 요청마다 1초가 지났는지 확인
        if requests_made % 20 == 0:
            elapsed_time = time.time() - start_time_1s
            if elapsed_time < 2:
                time.sleep(2 - elapsed_time)
            start_time_1s = time.time()

        # 100번 요청마다 약 2분이 지났는지 확인
        if requests_made % 100 == 0:
            elapsed_time = time.time() - start_time_2m
            if elapsed_time < 125:
                time.sleep(125 - elapsed_time)
            start_time_2m = time.time()

    except requests.exceptions.RequestException as e:
        print(f"소환사 ID {puuid}에서 데이터 가져오기 오류: {e}")
        continue

    # 진행 상황 출력
    if (i + 1) % 10 == 0 or (i + 1) == total_ids:
        print(f"{i + 1}/{total_ids} 개의 소환사 정보 처리 완료.")

# 모든 데이터 수집 후 JSON 파일로 저장
with open('champion_mastery_top.json', 'w', encoding='utf-8') as f:
    json.dump(summoner_data_dict, f, ensure_ascii=False, indent=4)


print("전체 데이터 수집 완료")