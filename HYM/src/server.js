const app = require("./app");
const connectDB = require("./config/db");

const PORT = process.env.PORT || 8080;
const MONGO_URI = "mongodb://localhost:27017/TFT_data";

// MongoDB 연결 후 서버 실행
connectDB(MONGO_URI).then(() => {
  app.listen(PORT, () => {
    console.log(`✅ 서버가 http://localhost:${PORT} 에서 실행 중입니다.`);
  });
});
