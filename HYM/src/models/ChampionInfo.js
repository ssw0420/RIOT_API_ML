const mongoose = require("mongoose");

const ChampionInfoSchema = new mongoose.Schema({
  championId: Number, // 챔피언 ID
  championName: String, // 챔피언 이름
  img_url: String, // 이미지 URL
});

module.exports = mongoose.model("ChampionInfo", ChampionInfoSchema, "champion_infos_eg");
