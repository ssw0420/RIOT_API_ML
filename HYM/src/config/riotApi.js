const axios = require("axios");

const RIOT_API_KEY = "RGAPI-1c45b387-646a-4cc6-ade9-7e6959f279bc"; // Riot API 키
const RIOT_API_HEADERS = {
  "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
  "Accept-Language": "ko-KR,ko;q=0.9,en-GB;q=0.8,en-US;q=0.7,en;q=0.6",
  "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
  Origin: "https://developer.riotgames.com",
  "X-Riot-Token": RIOT_API_KEY,
};

// Riot API를 사용하여 PUUID 추출 함수
const fetchPUUID = async (nickname, tag) => {
  try {
    const region = "asia"; // 아시아 지역 클러스터
    const response = await axios.get(`https://${region}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/${encodeURIComponent(nickname)}/${encodeURIComponent(tag)}`, {
      headers: RIOT_API_HEADERS,
    });

    const { puuid } = response.data;
    return puuid;
  } catch (err) {
    console.error("PUUID 추출 중 오류 발생:", err.response?.data || err.message);
    throw new Error("PUUID를 가져오는 중 오류가 발생했습니다.");
  }
};

// 최고 숙련도 챔피언 3개의 championId 추출 함수
const fetchTopChampions = async (region, puuid) => {
  try {
    const response = await axios.get(`https://${region}.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-puuid/${encodeURIComponent(puuid)}/top?count=3`, { headers: RIOT_API_HEADERS });

    // championId만 추출하여 배열로 반환
    return response.data.map((champion) => champion.championId);
  } catch (err) {
    console.error("최고 숙련도 챔피언 데이터 호출 중 오류 발생:", err.response?.data || err.message);
    throw new Error("최고 숙련도 챔피언 데이터를 가져오는 중 오류가 발생했습니다.");
  }
};

module.exports = {
  fetchPUUID,
  fetchTopChampions,
};
