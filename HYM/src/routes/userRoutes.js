const express = require("express");
const UserInfo = require("../models/UserInfo");
const { fetchPUUID, fetchTopChampions } = require("../config/riotApi");

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

    // MongoDB에 데이터 저장 또는 업데이트
    const existingUser = await UserInfo.findOne({ nickname, tag });

    if (existingUser) {
      // 기존 데이터 업데이트
      existingUser.name = name;
      existingUser.puuid = puuid;
      existingUser.topChampions = topChampions; // championId와 championPoints 저장
      await existingUser.save();

      return res.status(200).json({
        message: "기존 사용자 정보가 업데이트되었습니다.",
        user: existingUser,
      });
    }

    // 새 데이터 저장
    const newUser = new UserInfo({
      name,
      nickname,
      tag,
      puuid,
      topChampions, // championId와 championPoints 저장
    });
    const savedUser = await newUser.save();

    res.status(201).json({
      message: "사용자 정보와 최고 숙련도 챔피언 데이터가 저장되었습니다.",
      user: savedUser,
    });
  } catch (err) {
    console.error("사용자 저장 중 오류 발생:", err.message);
    res.status(500).json({ message: "서버 오류 발생." });
  }
});

module.exports = router;
