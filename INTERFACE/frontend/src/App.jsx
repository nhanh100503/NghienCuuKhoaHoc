import { Routes, Route } from "react-router-dom"
import HomePage from "./pages/HomePage"
import ClassificationPage from "./pages/ClassificationPage"
import SimilarityPage from "./pages/SimilarityPage"
import PdfSimilarityPage from "./pages/PdfSimilarityPage"

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/classification" element={<ClassificationPage />} />
        <Route path="/similarity" element={<SimilarityPage />} />
        <Route path="/pdf-similarity" element={<PdfSimilarityPage />} />
      </Routes>
    </div>
  )
}

export default App