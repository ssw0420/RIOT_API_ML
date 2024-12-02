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

# REQUEST_HEADERS = {
#     "X-Riot-Token": api_key
# }

userNickname='자크빼면시체'
tagLine='KR1'
# encodedName = parse.quote(userNickname)
# print(encodedName)
puuid = 'kQxfpLmp3R4QPfIVqE5Yh88ZV48h_zHGBfUR_ElJF7JR_MY_5jWJeYXn1yJkhN4_3-NUbSAo3MKNKA'
url = f"https://asia.api.riotgames.com/riot/account/v1/accounts/by-puuid/{puuid}"

player_id = requests.get(url, headers=REQUEST_HEADERS).json()

print(player_id)