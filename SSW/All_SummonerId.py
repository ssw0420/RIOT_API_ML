import requests
import json
from urllib import parse
from dotenv import load_dotenv
import os

# json_file_names = ['challengerleagues.json', 'grandmasterleagues.json', 'masterleagues.json']
json_file_names = ['challengerleagues.json', 'grandmasterleagues.json']

# with open('challengerleagues.json', 'r') as file:
#      data_json = file.read()

# data = json.loads(data_json)
# challengerleagues_summoner_ids = [entry['summonerId'] for entry in data['entries']]

summoner_ids = []

for file_name in json_file_names:
    with open(file_name, 'r') as file:
        data = json.load(file)
        summoner_ids.extend([entry['summonerId'] for entry in data['entries']])

summoner_ids = list(set(summoner_ids))

with open('ch_gm_summoner_ids.json', 'w') as output_file:
    json.dump(summoner_ids, output_file, indent=4)

print("챌린저, 그랜드마스터 json 소환사 아이디 병합")