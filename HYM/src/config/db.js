const mongoose = require("mongoose");

const connectDB = async (MONGO_URI) => {
  try {
    await mongoose.connect(MONGO_URI);
    console.log("✅ MongoDB 연결 성공!");
  } catch (err) {
    console.error("❌ MongoDB 연결 실패:", err.message);
    process.exit(1);
  }
};

module.exports = connectDB;
