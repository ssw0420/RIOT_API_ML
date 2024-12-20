const mongoose = require("mongoose");

const AssociationTextSchema = new mongoose.Schema({
  number: Number, // 클러스터 번호
  title: String, // 제목
  description: String, // 설명
  details: [String], // 세부 내용 (배열)
  note: String, // 추가 설명
  similar_pro_gamers: [String], // 유사 프로게이머 (배열)
});

module.exports = mongoose.model("association_text", AssociationTextSchema, "association_text");
