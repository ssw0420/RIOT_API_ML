const express = require("express");
const UserInfo = require("../models/UserInfo");
const ChampionInfo = require("../models/championInfo");
const { fetchPUUID, fetchTopChampions } = require("../config/riotApi");
const AssociationCard = require("../models/AssociationCard");
const AssociationText = require("../models/AssociationText");

const axios = require("axios");

const router = express.Router();

// 사용자 정보 저장 및 업데이트 API
router.post("/saveUser", async (req, res) => {
  const { name, nicknameTag } = req.body;

  // 닉네임과 태그 분리
  const [nickname, tag] = nicknameTag.split("#");
  if (!nickname || !tag) {
    return res.status(400).json({ message: "닉네임#태그 형식을 확인하세요!" });
  }

  try {
    // Riot API를 통해 PUUID 추출
    console.log(`Riot API를 호출하여 PUUID를 가져오는 중: ${nickname}#${tag}`);
    const puuid = await fetchPUUID(nickname, tag);

    // 최고 숙련도 챔피언 데이터 가져오기
    console.log(`PUUID를 통해 최고 숙련도 챔피언 데이터를 가져오는 중: ${puuid}`);
    const topChampions = await fetchTopChampions("kr", puuid); // 한국 서버 예시

    // MongoDB에서 사용자 조회
    let user = await UserInfo.findOne({ nickname, tag });

    // Python 서버로 데이터 전송
    console.log("Python 서버로 데이터 전송 중...");
    const pythonResponse = await axios.post("http://localhost:5000/send-data", {
      name,
      nickname,
      tag,
      puuid,
      topChampions,
    });

    console.log("Python 서버 응답:", pythonResponse.data);
    const assignedCluster = pythonResponse.data.assignedCluster; // 클러스터 할당 결과

    if (user) {
      // 기존 사용자 정보 업데이트
      user.name = name;
      user.puuid = puuid;
      user.topChampions = topChampions;
      user.assignedCluster = assignedCluster; // 클러스터 정보 저장
      await user.save();
      console.log("MongoDB에서 사용자 데이터 업데이트 완료");
    } else {
      // 신규 사용자 정보 저장
      user = new UserInfo({
        name,
        nickname,
        tag,
        puuid,
        topChampions,
        assignedCluster, // 신규 사용자 문서에 클러스터 정보 저장
      });
      await user.save();
      console.log("MongoDB에 사용자 데이터 저장 완료");
    }

    // 클라이언트로 결과 반환
    res.json({
      user: user,
      status: "success",
      pythonResult: pythonResponse.data,
    });
  } catch (err) {
    console.error("오류 발생:", err.message);
    res.status(500).json({ message: "서버 오류 발생.", error: err.message });
  }
});

// 사용자 정보로 이미지 URL 가져오기 API
router.post("/getChampionImage", async (req, res) => {
  const { name, nickname, tag } = req.body;

  try {
    // `user_infos` 컬렉션에서 사용자 정보 검색
    const user = await UserInfo.findOne({ name, nickname, tag });
    if (!user) {
      return res.status(404).json({ message: "사용자를 찾을 수 없습니다." });
    }

    // 사용자의 topChampions 배열에서 첫 번째 챔피언 ID 가져오기
    const firstChampionId = user.topChampions[0]?.championId;
    if (!firstChampionId) {
      return res.status(400).json({ message: "사용자에게 등록된 챔피언 정보가 없습니다." });
    }

    // `champion_infos_eg` 컬렉션에서 championId로 이미지 URL 검색
    const championInfo = await ChampionInfo.findOne({ championId: firstChampionId });
    if (!championInfo) {
      return res.status(404).json({ message: "챔피언 정보를 찾을 수 없습니다." });
    }

    // 이미지 URL 반환
    return res.status(200).json({ img_url: championInfo.img_url });
  } catch (error) {
    console.error("오류 발생:", error.message);
    return res.status(500).json({ message: "서버 오류 발생", error: error.message });
  }
});

// 클러스터에 맞는 카드 데이터 불러오기 API
router.post("/getCardData", async (req, res) => {
  const { name, nickname, tag } = req.body;

  try {
    // user_infos 컬렉션에서 사용자 데이터 조회
    const user = await UserInfo.findOne({ name, nickname, tag });

    if (!user) {
      return res.status(404).json({ message: "사용자 정보를 찾을 수 없습니다." });
    }

    const assignedCluster = user.assignedCluster;

    // association_card 컬렉션에서 클러스터 번호와 일치하는 데이터 조회
    const cardData = await AssociationCard.findOne({ number: assignedCluster });

    if (!cardData) {
      return res.status(404).json({ message: "클러스터에 해당하는 카드 데이터를 찾을 수 없습니다." });
    }

    res.status(200).json({
      card_title: cardData.card_title,
      icon_content: cardData.icon_content,
    });
  } catch (error) {
    console.error("오류 발생:", error.message);
    res.status(500).json({ message: "서버 오류 발생.", error: error.message });
  }
});

// association_text에서 데이터 불러오기 API
router.post("/getTextData", async (req, res) => {
  const { name, nickname, tag } = req.body;

  try {
    // user_infos 컬렉션에서 사용자 데이터 조회
    const user = await UserInfo.findOne({ name, nickname, tag });

    if (!user) {
      return res.status(404).json({ message: "사용자 정보를 찾을 수 없습니다." });
    }

    const assignedCluster = user.assignedCluster;

    // association_text 컬렉션에서 클러스터 번호와 일치하는 데이터 조회
    const textData = await AssociationText.findOne({ number: assignedCluster });

    if (!textData) {
      return res.status(404).json({ message: "클러스터에 해당하는 텍스트 데이터를 찾을 수 없습니다." });
    }

    res.status(200).json({
      title: textData.title,
      description: textData.description,
      details: textData.details,
      note: textData.note,
      similar_pro_gamers: textData.similar_pro_gamers,
    });
  } catch (error) {
    console.error("오류 발생:", error.message);
    res.status(500).json({ message: "서버 오류 발생.", error: error.message });
  }
});

module.exports = router;
