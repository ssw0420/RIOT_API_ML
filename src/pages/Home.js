import React, { useState } from "react";
import Background from "../components/Background";
import "../styles/Home.css";
import MainChamp from "../assets/main_champ.png";
import BlurredChamp from "../assets/blur_champ.png";

const Home = () => {
  const [isBlurred, setIsBlurred] = useState(false);

  const handleShowResults = () => {
    setIsBlurred(true);
  };

  const handleBackToHome = () => {
    setIsBlurred(false);
  };

  return (
    <Background>
      <div className="container">
        <div className="wrap">
          <div className="image-container">
            <img
              src={isBlurred ? BlurredChamp : MainChamp}
              alt="Main Champion"
              className={`main-champ`}
            />
            {isBlurred && (
              <div className="overlay-content fade-in">
                <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
                <input type="text" placeholder="이름" className="input-field" />
                <input
                  type="text"
                  placeholder="소환사명#태그"
                  className="input-field"
                />
                <button className="button">결과 보기</button>
              </div>
            )}
            <div
              className={`main-text-div ${
                isBlurred ? "blurred-text" : ""
              }`}
            >
              <h1>
                LOL을 통해 보는<br />나의 TFT 성향은?
              </h1>
              <button
                className={`button ${isBlurred ? "blurred-button" : ""}`}
                onClick={isBlurred ? handleBackToHome : handleShowResults}
              >
                {isBlurred ? "홈으로 이동" : "성향 알아보기"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </Background>
  );
};

export default Home;
