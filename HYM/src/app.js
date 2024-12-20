const express = require("express");
const bodyParser = require("body-parser");
const path = require("path");
const userRoutes = require("./routes/userRoutes");

const app = express();

// Middleware 설정
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, "../public"))); // 정적 파일 제공

// 라우트 설정
app.use("/api", userRoutes);

// 기본 경로 설정
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "../public/home.html"));
});

module.exports = app;
