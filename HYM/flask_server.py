from flask import Flask, request, jsonify

app = Flask(__name__)

# Node.js에서 데이터를 수신하고 처리
@app.route('/process-data', methods=['POST'])
def process_data():
    try:
        # Node.js에서 보낸 JSON 데이터 받기
        data = request.json
        print("받은 데이터:", data)

        # 데이터 처리 로직 (여기서는 테스트로 가공된 데이터를 반환)
        processed_data = {"status": "success", "message": "데이터 처리 완료", "result": data}

        # 처리 결과를 반환
        return jsonify(processed_data), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Flask 서버 실행
