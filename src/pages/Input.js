import React from "react";
import Background from "../components/Background";
import BlurredChamp from "../assets/blur_champ.png";
import "../styles/Input.css";

const Input = () => {
  return (
    <Background>
      <div className="container">
        <div className="wrap">
          <div className="image-container">
            <img src={BlurredChamp} alt="Main Champion" className="main-champ" />
            <div className="main-text-div">
              <h1>LOL을 통해 보는<br />나의 TFT 성향은?</h1>
              <button className="button">성향 알아보기</button>
            </div>
          </div>
        </div>
      </div>
    </Background>
  );
};

export default Input;
