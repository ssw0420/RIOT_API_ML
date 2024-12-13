// UserInfo.js
const mongoose = require("mongoose");

const UserInfoSchema = new mongoose.Schema({
  name: String,
  nickname: String,
  tag: String,
  puuid: { type: String, default: null },
  topChampions: [
    {
      championId: Number,
      championPoints: Number,
    },
  ],
  // 클러스터 할당 결과 저장 필드 추가
  assignedCluster: { type: String, default: null },
});

module.exports = mongoose.model("user_infos", UserInfoSchema);
//module.exports = mongoose.model("proGamer_infos", UserInfoSchema, "proGamer_infos");
