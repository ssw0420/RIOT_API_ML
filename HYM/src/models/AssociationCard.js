const mongoose = require("mongoose");

const AssociationCardSchema = new mongoose.Schema({
  number: Number, // 클러스터 번호
  card_title: String, // 카드 제목
  icon_content: [String], // 아이콘 내용 배열
});

module.exports = mongoose.model("association_card", AssociationCardSchema, "association_card");
