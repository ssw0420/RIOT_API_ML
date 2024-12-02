import React from "react";
import RiotLogo from "../assets/logo.png";

const styles = {
  logo: {
    position: "absolute",
    top: "20px",
    left: "40px",
    width: "90px",
    zIndex: 2,
  },
};

const Logo = () => {
  return (
    <img
      src={RiotLogo}
      alt="Riot Games Logo"
      style={styles.logo}
    />
  );
};

export default Logo;
