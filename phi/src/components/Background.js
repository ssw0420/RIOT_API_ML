import React from "react";
import backgroundImage from "../assets/background.png";
import Logo from "./Logo";

const styles = {
  container: {
    position: "relative",
    width: "100%",
    height: "100vh",
    overflow: "hidden",
  },
  image: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    objectFit: "fill",
    zIndex: -1,
  },
};

const Background = ({ children }) => {
  return (
    <div style={styles.container}>
      <Logo />
      <img
        src={backgroundImage}
        alt="Background"
        style={styles.image}
      />
      {children}
    </div>
  );
};

export default Background;
