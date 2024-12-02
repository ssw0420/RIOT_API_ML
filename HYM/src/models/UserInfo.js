const mongoose = require("mongoose");

const UserInfoSchema = new mongoose.Schema({
  name: String, // 사용자 이름
  nickname: String, // 소환사 닉네임
  tag: String, // 소환사 태그
  puuid: { type: String, default: null }, // Riot API에서 가져온 PUUID
  topChampions: [Number], // 최고 숙련도 챔피언 ID 리스트
});

module.exports = mongoose.model("user_infos", UserInfoSchema);
