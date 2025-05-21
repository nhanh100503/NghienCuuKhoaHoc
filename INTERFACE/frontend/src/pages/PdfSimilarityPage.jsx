import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { Card, CardContent } from "../components/Card";
import Button from "../components/Button";
import PopupDetails from "../components/PopupDetail";

function PdfSimilarityPage() {
  const [sourcePdf, setSourcePdf] = useState(null); // Stores the file and name
  const [pdfUrl, setPdfUrl] = useState(null); // Stores the URL for iframe
  const [similarImages, setSimilarImages] = useState([]);
  const [predictedClass, setPredictedClass] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelName, setModelName] = useState("MobileNetV2");
  const [threshold, setThreshold] = useState(0.95);
  const [showPopup, setShowPopup] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [totalImages, setTotalImages] = useState(0);

  const handleSourcePdfSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type === "application/pdf") {
      const url = URL.createObjectURL(file); // Create a URL for the PDF file
      setSourcePdf({ file, name: file.name });
      setPdfUrl(url); // Store the URL for iframe
      setSimilarImages([]);
      setPredictedClass(null);
      setError(null);
    } else {
      setError("Vui lòng chọn file PDF hợp lệ");
      setSourcePdf(null);
      setPdfUrl(null);
    }
  };

  const handleSearchSimilar = async () => {
    if (!sourcePdf) {
      setError("Vui lòng chọn file PDF nguồn");
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("pdf", sourcePdf.file);
    formData.append("model_name", modelName);
    formData.append("threshold", threshold);

    try {
      const response = await fetch("http://localhost:5001/similarity", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();

      setPredictedClass(data.predicted_class);
      setSimilarImages(data.similar_images || []);
      setTotalImages(data.total_similar_images || 0);
    } catch (err) {
      setError(err.message || "Có lỗi xảy ra khi tìm kiếm ảnh tương tự");
    } finally {
      setIsLoading(false);
    }
  };

  const clearSourcePdf = () => {
    if (pdfUrl) {
      URL.revokeObjectURL(pdfUrl); // Clean up the URL to free memory
    }
    setSourcePdf(null);
    setPdfUrl(null);
    setSimilarImages([]);
    setTotalImages(0);
    setPredictedClass(null);
    setError(null);
  };

  const handleShowImageDetail = (image) => {
    setSelectedImage(image);
    setShowPopup(true);
    console.log("Selected image:", image);
  };

  // Clean up URL when component unmounts
  useEffect(() => {
    return () => {
      if (pdfUrl) {
        URL.revokeObjectURL(pdfUrl); // Ensure memory is freed on unmount
      }
    };
  }, [pdfUrl]);

  return (
    <div className="flex flex-col h-screen p-6">
      <div className="mb-6">
        <Link to="/">
          <Button variant="outline" size="sm" className="flex items-center gap-1">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-4 w-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M10 19l-7-7m0 0l7-7m-7 7h18"
              />
            </svg>
            Back to home
          </Button>
        </Link>
      </div>

      <h1 className="text-2xl font-bold mb-8 text-center">Similarity Computation</h1>

      <div className="grid grid-cols-1 md:grid-cols-[2fr_3fr] gap-8 mb-8">
        <div className="flex flex-col gap-4">
          <Card className="cursor-pointer bg-white border border-gray-200 rounded-lg shadow-sm">
            <CardContent>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-medium">Upload PDF</h2>
                {sourcePdf && (
                  <button
                    onClick={clearSourcePdf}
                    className="flex items-center gap-1 bg-red-500 hover:bg-red-600 text-white rounded-md px-3 py-1.5 text-sm transition-all duration-200 shadow-sm hover:shadow-md active:bg-red-700"
                    title="Xóa PDF"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-4 w-4"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                      />
                    </svg>
                    <span>Clear</span>
                  </button>
                )}
              </div>
              <div className="flex flex-col gap-4">
                <div className="relative h-64">
                  {sourcePdf && pdfUrl ? (
                    <div className="relative h-full overflow-auto">
                      <iframe
                        src={pdfUrl}
                        width="100%"
                        height="100%"
                        title="PDF Preview"
                        style={{ border: "none" }}
                      />
                    </div>
                  ) : (
                    <div className="h-full flex flex-col items-center justify-center">
                      <input
                        type="file"
                        accept="application/pdf"
                        onChange={handleSourcePdfSelect}
                        className="hidden"
                        id="pdf-upload"
                      />
                      <label
                        htmlFor="pdf-upload"
                        className="cursor-pointer flex flex-col items-center justify-center w-full h-full border-2 border-dashed border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                      >
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className="h-12 w-12 text-gray-400 mb-2"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                          />
                        </svg>
                        <p className="text-gray-500 text-center">Drag and drop PDF here or click to select</p>
                      </label>
                    </div>
                  )}
                </div>
                {sourcePdf ? (
                  <p className="text-sm text-gray-700">
                    <strong>Selected: </strong> {sourcePdf.name}
                  </p>
                ) : (
                  <p className="text-sm text-gray-700">No PDF selected</p>
                )}
              </div>
            </CardContent>
          </Card>
          <div className="flex flex-col md:flex-row gap-4 justify-center items-center mb-4">
            <div className="relative w-full md:w-3/4">
              <label className="block mb-1 font-medium">Select model</label>
              <select
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                className="w-full appearance-none py-2 px-3 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white pr-8"
              >
                <option value="MobileNetV2">MobileNetV2</option>
                <option value="ResNet101">ResNet101</option>
                <option value="EfficientNetB0">EfficientNetB0</option>
              </select>
              <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700 pt-6">
                <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                  <path
                    fillRule="evenodd"
                    d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
            </div>
            <div className="relative w-full md:w-1/4">
              <label className="block mb-1 font-medium">Threshold (0 - 1)</label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value) || 0)} // Ensure threshold is a number
                className="border border-gray-300 rounded px-3 py-2 w-full"
              />
            </div>
          </div>
          <Button
            onClick={handleSearchSimilar}
            disabled={!sourcePdf || isLoading}
            variant="sky"
          >
            {isLoading ? "Processing ..." : "Submit"}
          </Button>

          {error && <p className="text-red-500 text-center">{error}</p>}
        </div>

        <Card className="bg-white border border-gray-200 rounded-lg shadow-sm h-[533px] overflow-hidden">
          <CardContent className="p-4 h-full flex flex-col">
            <h2 className="text-lg font-medium mb-4">Result</h2>
            {isLoading ? (
              <div className="text-center pt-50">
                <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
              </div>
            ) : predictedClass ? (
              <div className="text-center mb-4">
                <p className="text-gray-600">
                  Classification results: <span className="font-bold">{predictedClass}</span>
                </p>
                <p className="text-gray-600">
                  Total similar images: <span className="font-bold">{totalImages}</span>
                </p>
              </div>
            ) : (
              <p className="text-center text-gray-600 pt-50">
                Please select a PDF to calculate image similarity
              </p>
            )}
            <div className="flex-1 overflow-auto">
              {similarImages.length > 0 && !isLoading ? (
                <div>
                  <div className="space-y-4">
                    {similarImages.map((image, index) => (
                      <Card key={index} className="border border-gray-200 rounded-lg">
                        <CardContent className="flex items-center p-4">
                          <div className="w-32 h-32 flex-shrink-0">
                            <img
                              src={image.image_data || "/placeholder.svg"}
                              alt={image.image_name}
                              className="w-full h-full object-contain"
                              onError={(e) => { e.target.src = "/placeholder.svg"; }}
                            />
                          </div>
                          <div className="ml-4 text-sm text-gray-600 space-y-2">
                            <p><strong>Similarity: </strong>{(image.similarity * 100).toFixed(2)}%</p>
                            <p><strong>Image name: </strong>{image.image_name}</p>
                            <p><strong>Title: </strong>{image.title}</p>
                            <p><strong>Caption: </strong>{image.caption}</p>
                            <p><strong>Authors: </strong>{image.authors}</p>
                            <div className="flex justify-between items-center mt-2">
                              <p className="text-sm text-right">
                                <strong>DOI: </strong>
                                {image.doi ? (
                                  <a
                                    href={`https://doi.org/${image.doi}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-blue-500 hover:underline"
                                  >
                                    {image.doi}
                                  </a>
                                ) : (
                                  "N/A"
                                )}
                              </p>
                              <p
                                className="cursor-pointer text-blue-500 hover:underline text-sm"
                                onClick={() => handleShowImageDetail(image)}
                              >
                                View detail
                              </p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>
              ) : !isLoading && predictedClass ? (
                <p className="text-center text-gray-600">
                  No similar images found with current threshold.
                </p>
              ) : null}
            </div>
          </CardContent>
        </Card>
      </div>
      {showPopup && sourcePdf && selectedImage && (
        <PopupDetails
          originalImage={{
            image_data: null,
            name: sourcePdf.name,
            caption: similarImages.find(img => img.image_id === selectedImage.image_id)?.caption || "N/A",
            doi: similarImages.find(img => img.image_id === selectedImage.image_id)?.doi,
          }}
          similarImage={selectedImage}
          onClose={() => setShowPopup(false)}
        />
      )}
    </div>
  );
}

export default PdfSimilarityPage;