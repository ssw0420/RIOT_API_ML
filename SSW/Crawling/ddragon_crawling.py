import json
import os
import requests
from urllib.parse import quote

def download_champion_images(json_file: str, output_dir: str, version: str):
    """
    주어진 JSON 파일에서 챔피언 이름을 추출하여 이미지 URL에 접근하고 이미지를 다운로드합니다.

    :param json_file: 챔피언 정보를 담은 JSON 파일 경로
    :param output_dir: 이미지를 저장할 디렉토리 경로
    :param version: 리그 오브 레전드의 데이터 버전 (예: '14.24.1')
    """
    # URL 템플릿
    url_template = f"https://ddragon.leagueoflegends.com/cdn/{version}/img/champion/{{}}.png"

    # 출력 디렉토리 생성 (없을 경우)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"디렉토리를 생성했습니다: {output_dir}")

    try:
        # JSON 데이터 로드
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # "data" 키가 아닌 최상위 레벨에서 챔피언 정보 가져오기
        champions = data  # champions_filtered.json의 구조에 맞게 변경

        if not champions:
            print("챔피언 데이터가 비어 있습니다. JSON 파일의 구조를 확인해주세요.")
            return

        total = len(champions)
        print(f"총 {total}명의 챔피언 이미지를 다운로드합니다.")

        for idx, (champ_name, champ_info) in enumerate(champions.items(), start=1):
            # 챔피언 이름을 URL에 맞게 인코딩 (특수 문자 처리)
            encoded_name = quote(champ_name)
            image_url = url_template.format(encoded_name)

            # 이미지 저장 경로
            image_path = os.path.join(output_dir, f"{champ_name}.png")

            try:
                # HTTP GET 요청
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()  # HTTP 에러 발생 시 예외 발생

                # 이미지 파일 저장
                with open(image_path, 'wb') as img_file:
                    img_file.write(response.content)

                print(f"[{idx}/{total}] 다운로드 성공: {champ_name}.png")

            except requests.exceptions.HTTPError as http_err:
                print(f"[{idx}/{total}] HTTP 에러 발생 ({image_url}): {http_err}")
            except requests.exceptions.ConnectionError as conn_err:
                print(f"[{idx}/{total}] 연결 에러 발생 ({image_url}): {conn_err}")
            except requests.exceptions.Timeout as timeout_err:
                print(f"[{idx}/{total}] 타임아웃 에러 발생 ({image_url}): {timeout_err}")
            except requests.exceptions.RequestException as req_err:
                print(f"[{idx}/{total}] 요청 에러 발생 ({image_url}): {req_err}")
            except Exception as e:
                print(f"[{idx}/{total}] 예외 발생 ({image_url}): {e}")

        print("모든 이미지 다운로드가 완료되었습니다.")

    except FileNotFoundError:
        print(f"에러: 파일을 찾을 수 없습니다 - {json_file}")
    except json.JSONDecodeError:
        print("에러: JSON 파일의 형식이 올바르지 않습니다.")
    except Exception as e:
        print(f"예상치 못한 에러 발생: {e}")

if __name__ == "__main__":
    # 입력 및 출력 파일 경로 설정
    input_json = 'Crawling\champions_filtered.json'       # 원본 JSON 파일 경로
    output_directory = 'champion_images'         # 이미지 저장 디렉토리
    game_version = '14.24.1'                     # 데이터 버전

    # 함수 호출
    download_champion_images(input_json, output_directory, game_version)
