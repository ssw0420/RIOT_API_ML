import requests
import json
from urllib import parse
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('RIOT_API_KEY')
print(api_key)

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "https://developer.riotgames.com",
    "X-Riot-Token": api_key
}


queue = 'RANKED_SOLO_5x5'

lol_grandmasterleagues_url = f"https://kr.api.riotgames.com/lol/league/v4/grandmasterleagues/by-queue/{queue}"

lol_grandmasterleagues_json = requests.get(lol_grandmasterleagues_url, headers=REQUEST_HEADERS).json()

print("API 호출 완료")

with open('grandmasterleagues.json', 'w') as f:
    json.dump(lol_grandmasterleagues_json, f, indent=4)

print("json 파일 작성 완료")