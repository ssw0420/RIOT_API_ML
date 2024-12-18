const express = require("express");
const UserInfo = require("../models/UserInfo");
const { fetchPUUID, fetchTopChampions } = require("../config/riotApi");
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

    // MongoDB에 데이터 저장 또는 업데이트
    const existingUser = await UserInfo.findOne({ nickname, tag });

    if (existingUser) {
      // 기존 데이터 업데이트
      existingUser.name = name;
      existingUser.puuid = puuid;
      existingUser.topChampions = topChampions; // championId와 championPoints 저장
      await existingUser.save();
      console.log("MongoDB에서 사용자 데이터 업데이트 완료");
    } else {
      // 새 데이터 저장
      const newUser = new UserInfo({
        name,
        nickname,
        tag,
        puuid,
        topChampions, // championId와 championPoints 저장
      });
      await newUser.save();
      console.log("MongoDB에 사용자 데이터 저장 완료");
    }

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

    // 클라이언트로 결과 반환
    res.json({
      status: "success",
      message: "사용자 정보 저장 및 Python 처리 완료",
      pythonResult: pythonResponse.data,
    });
  } catch (err) {
    console.error("오류 발생:", err.message);
    res.status(500).json({ message: "서버 오류 발생.", error: err.message });
  }
});

module.exports = router;
