from flask import Flask, request, jsonify

app = Flask(__name__)

# 샘플 API 엔드포인트
@app.route('/send-data', methods=['POST'])
def send_data():
    # JSON 데이터 받기
    data = request.json
    print("받은 데이터:", data)

    # 응답 반환
    return jsonify({"status": "success", "message": "데이터가 성공적으로 처리되었습니다!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # 서버 실행
