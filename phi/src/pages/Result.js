import React from "react";
import Background from "../components/Background";
import BgTitle from "../assets/bgTitle.png";
import Icon1 from "../assets/card_Icon1.png";
import Icon2 from "../assets/card_Icon2.png";
import Icon3 from "../assets/card_Icon3.png";
import "../styles/Result.css";

const Result = () => {
  return (
    <Background>
      <div className="container">
        <div className="wrap">
          <div className="title-area">
            <h1>해인님의 전략적 팀 전투 성향은...</h1>
          </div>
          <div className="content-area">
            <div className="card">
              <img src="https://ddragon.leagueoflegends.com/cdn/img/champion/loading/Aatrox_0.jpg" alt="전략 카드" className="card-image" />
              <img src={BgTitle} alt="배경 타이틀" className="bg-title-image" />
              <p className="card-title">밸런스 중심</p>
              <div className="card-footer">
                <p id="card-player">Oner#KR222</p>
                <div className="icon-content">
                  <img src={Icon1} alt="Icon 1" className="footer-icon" />
                  <p>다양한 조합과 밸런스를 중시하며, 특정 전략에 지나치게 의존하지 않는 특성</p>
                </div>
                <div className="icon-content">
                  <img src={Icon2} alt="Icon 2" className="footer-icon" />
                  <p>다양한 조합과 밸런스를 중시하며, 특정 전략에 지나치게 의존하지 않는 특성</p>
                </div>
                <div className="icon-content">
                  <img src={Icon3} alt="Icon 3" className="footer-icon" />
                  <p>다양한 조합과 밸런스를 중시하며, 특정 전략에 지나치게 의존하지 않는 특성</p>
                </div>
              </div>
            </div>

            <div className="description">
              <h2>T1 Faker의 전략적 밸런스 플레이를 닮은 플레이 스타일</h2>
              <div className="content">
                <p>이 플레이 스타일은 검은 장미단을 중심으로 다양한 조합과 전략을 통해 안정성과 유연성을 동시에 추구하는 것이 특징입니다.</p>
                <ul>
                  <li>검은 장미단을 메인 축으로 하여 안정적인 초반 운영</li>
                  <li>투사, 매복자, 반군 등 다양한 시너지를 활용하여 유연한 전투 설계</li>
                  <li>상대의 전략을 파악하고 과감한 결단으로 승기를 잡는 순간을 창출</li>
                  <li>초반부터 중후반까지 고른 성능을 유지하여 게임 전체의 흐름을 주도</li>
                </ul>
                <p>
                  이러한 플레이 스타일은 T1 Faker와 유사하게 뛰어난 판단력과 섬세한 운영 능력을 겸비한 플레이어들에게서 자주 나타나며, 팀과의 조화로 게임을 장악하는 데 있어 탁월한 능력을 발휘합니다.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Background>
  );
};

export default Result;
