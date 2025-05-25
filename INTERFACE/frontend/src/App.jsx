import { Routes, Route } from "react-router-dom";
import HomePage from "./pages/HomePage";
import SimilarityPage from "./pages/SimilarityPage";
import PdfSimilarityPage from "./pages/PdfSimilarityPage";
import SimilarityPdfPage from "./pages/SimilarityPdfPage";
function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/similarity" element={<SimilarityPage />} />
        <Route path="/pdf-similarity" element={<PdfSimilarityPage />} />
        <Route path="/pdf/similarity-images" element={<SimilarityPdfPage />} />
      </Routes>
    </div>
  );
}

export default App;
