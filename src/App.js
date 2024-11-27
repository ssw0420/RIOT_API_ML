import './App.css';
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from './pages/Home.js';
import Input from './pages/Input.js';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/input" element={<Input />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
