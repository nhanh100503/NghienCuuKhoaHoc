import React, { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Card, CardContent } from "../components/Card";
import Button from "../components/Button";
import PopupDetails from "../components/PopupDetail";
import ExtractedService from "../services/Extracted.service";
import axios from "axios";
import UploadPdf from "../components/UploadPdf";
import { useLocation } from "react-router-dom";
import { toast, ToastContainer } from "react-toastify";
import { Alert, Flex, Spin } from "antd";
import { FaCheck } from "react-icons/fa";
import { CiBoxList } from "react-icons/ci";
import { CiTrash } from "react-icons/ci";
import { FaThList } from "react-icons/fa";
import { CiImageOn } from "react-icons/ci";

function PdfSimilarityPage() {
  const [sourcePdf, setSourcePdf] = useState(null); // Stores the file and name
  const [pdfUrl, setPdfUrl] = useState(null); // Stores the URL for iframe
  const [similarImages, setSimilarImages] = useState([]);
  const [predictedClass, setPredictedClass] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelName, setModelName] = useState("ResNet101");
  const [threshold, setThreshold] = useState(0.8);
  const [showPopup, setShowPopup] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [totalImages, setTotalImages] = useState(0);
  const [extractedImages, setExtractedImages] = useState([]);
  const [chosenImages, setChosenImages] = useState([]);
  const [isSelectAll, setIsSelectAll] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const [isCloseDiv, setIsCloseDiv] = useState(false);
  const [isExtracted, setIsExtracted] = useState(false)
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

  const clearSourcePdf = () => {
    if (pdfUrl) {
      URL.revokeObjectURL(pdfUrl);
    }
    setSourcePdf(null);
    setPdfUrl(null);
    setSimilarImages([]);
    setTotalImages(0);
    setPredictedClass(null);
    setError(null);
    setExtractedImages([]);
    setSelectedImage(null)
    setIsSelectAll(false)
    setIsExtracted(false)
    setChosenImages([])
  };
  
  const handleChooseImages = (image) => {
    setChosenImages((prev) => {
      const alreadyChosen = prev.includes(image);
      if (alreadyChosen) {
        return prev.filter((img) => img !== image);
      } else {
        return [...prev, image]; 
      }
    });
  };
  const handleSelectAllImages = () => {
    if (isSelectAll) {
      setChosenImages([]);
    } else {
      setChosenImages([...extractedImages]);
    }
    setIsSelectAll(!isSelectAll);
  };

  const handleSearchSimilar = async () => {
    if (!sourcePdf) {
      setError("Vui lòng chọn file PDF nguồn");
      return;
    }

    if (chosenImages?.length == 0) {
      toast.warning("Please select at least one image");
      return;
    }
    setIsLoading(true);
    setError(null);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5001/similarity-pdf",
        {
          model_name: modelName,
          threshold: threshold,
          images: chosenImages,
        },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      console.log(response);
      navigate("/pdf/similarity-images", { state: response.data });
    } catch (err) {
      setError(err.message || "Có lỗi xảy ra khi tìm kiếm ảnh tương tự");
    }
    //  finally {
    //   setIsLoading(false);
    // }
  };

  const handleShowImageDetail = (image) => {
    setSelectedImage(image);
    setShowPopup(true);
  };
  const handleCloseDiv = () => {
    setIsCloseDiv(true);
  };

  const handleExtractImages = async () => {
    try {
      setIsLoading(true);
      const response = await ExtractedService.sendPdf({ pdf: sourcePdf.file });
      setExtractedImages(response);
      setIsExtracted(true)
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    return () => {
      if (pdfUrl) {
        URL.revokeObjectURL(pdfUrl); // Ensure memory is freed on unmount
      }
    };
  }, [pdfUrl]);
  console.log(extractedImages);
  return (
    <div className="flex flex-col h-screen">
      <ToastContainer position="top-right" autoClose={3000} />

      <div className="mb-6 flex items-center justify-between relative m-5">
        <div>
          <Link to="/">
            <Button
              variant="outline"
              size="sm"
              className="flex items-center gap-1"
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
                  d="M10 19l-7-7m0 0l7-7m-7 7h18"
                />
              </svg>
              Back to home
            </Button>
          </Link>
        </div>
        <div>
          <h1 className="text-3xl font-extrabold absolute top-0 left-1/2 transform -translate-x-1/2 text-sky-500 drop-shadow-md">
            Similarity Search for PDF Images
          </h1>
        </div>
        <div className="cursor-pointer">
          <Link to="/similarity">
            <CiImageOn className="w-7 h-7 text-sky-500" />
          </Link>
        </div>
      </div>

      <div
        className={`grid grid-cols-1  mb-2 px-6 ${
          isCloseDiv
            ? "md:grid-cols-[0.5fr_19fr] gap-4"
            : "md:grid-cols-[1.5fr_3fr] gap-8"
        }`}
      >
        {isCloseDiv ? (
          <div className=" h-[550px]  border rounded-lg border-gray-200 shadow  ">
            {extractedImages.length > 0 && (
              <div
                className="p-2 rounded-full hover:bg-sky-100 transition flex justify-end cursor-pointer"
                onClick={() => setIsCloseDiv(false)}
              >
                <CiBoxList className="w-6 h-6 text-sky-500" />
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-col h-[550px] m-0  border rounded-lg px-2 border-gray-200 shadow ">
            <div className="cursor-pointer border-none shadow-none ">
              <UploadPdf
                onClearPdf={clearSourcePdf}
                setPdfUrl={setPdfUrl}
                setSourcePdf={setSourcePdf}
                onSelectPdf={handleSourcePdfSelect}
                extractImages={extractedImages}
                pdfUrl={pdfUrl}
                sourcePdf={sourcePdf}
                onCloseDiv={handleCloseDiv}
              />
            </div>
            {pdfUrl && extractedImages.length > 0 && (
              <div className="flex flex-col justify-center items-center overflow-y-auto  ">
                <div className="text-sm text-gray-700 w-full px-4 ">
                  <p>
                    <strong>Title:</strong> {extractedImages[0].title}
                  </p>
                  <p>
                    <strong>Authors:</strong> {extractedImages[0].authors}
                  </p>
                  <p>
                    <strong>Accepted Date:</strong>{" "}
                    {extractedImages[0].approved_date}
                  </p>
                  <p>
                    <span className="font-semibold text-gray-800">DOI:</span>{" "}
                    <a
                      href={`https://doi.org/${extractedImages[0].doi}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-green-600 hover:underline break-words"
                    >
                      🔗 DOI:{" "}
                      {extractedImages[0].doi != null
                        ? extractedImages[0].doi
                        : "Not Found"}
                    </a>{" "}
                  </p>
                </div>
              </div>
            )}

            {error && <p className="text-red-500 text-center">{error}</p>}
          </div>
        )}
        {extractedImages.length > 0 && pdfUrl ? (
          <>
            <div>
              <div className="bg-white flex flex-col border overflow-auto border-gray-200 rounded-lg shadow-sm p-2 h-[550px]">
                <div className="flex justify-between items-center">
                  <h2 className="text-lg font-medium m-2">
                    Result: {extractedImages.length} images
                  </h2>

                  <div className="flex items-center mb-1 ">
                    <div
                      onClick={handleSelectAllImages}
                      className={`rounded-full hover:cursor-pointer shadow-xl hover:scale-105 mr-3 flex items-center justify-center
                        border border-gray-300 w-5 h-5  ring ring-white
                        transition-opacity duration-300 ${
                          isSelectAll ? "bg-blue-300" : "bg-gray-100"
                        } `}
                    >
                      {isSelectAll && <FaCheck className="text-white h-3 " />}
                    </div>
                    <span className="text-md text-gray-900">Select All</span>
                  </div>
                </div>
                <div
                  className={`grid gap-4 ${
                    isCloseDiv ? "grid-cols-6" : "grid-cols-4 "
                  }`}
                >
                  {extractedImages.map((image, index) => (
                    <div
                      key={index}
                      className="rounded-lg p-2 border border-gray-300 relative group hover:scale-105 hover:shadow-lg transition-transform duration-200"
                    >
                      <div
                        onClick={(e) => {
                          e.stopPropagation();
                          handleChooseImages(image);
                        }}
                        className={`rounded-full hover:cursor-pointer shadow-xl hover:scale-105 flex items-center justify-center
                        border border-gray-300 w-5 h-5 absolute -right-2 -top-2 ring ring-white
                        transition-opacity duration-300 
                        ${
                          chosenImages.includes(image)
                            ? "opacity-100 bg-blue-300"
                            : "opacity-0 group-hover:opacity-100 bg-gray-200"
                        }`}
                      >
                        {chosenImages.includes(image) && (
                          <FaCheck className="text-white h-3 " />
                        )}
                      </div>

                      <img
                        src={`data:image/png;base64,${image.base64}`}
                        alt=""
                        className="object-cover w-[300px] h-[200px]"
                      />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </>
        ) : (
          <Card className="bg-white border border-gray-200 rounded-lg shadow-sm overflow-hidden">
            <h2 className="text-lg font-medium  m-4">Result</h2>
            <CardContent className="p-4 flex flex-col">
              {isLoading ? (
                <div className="flex justify-center items-center w-full min-h-[300px]">
                  <Spin size="large" />
                </div>
              ) : predictedClass ? (
                <div className="text-center mb-4">
                  <p className="text-gray-600">
                    Classification results:{" "}
                    <span className="font-bold">{predictedClass}</span>
                  </p>
                  <p className="text-gray-600">
                    Total similar images:{" "}
                    <span className="font-bold">{totalImages}</span>
                  </p>
                </div>
              ) : (
                <p className="text-center text-gray-600 pt-50">
                  Please select a PDF to calculate image similarity
                </p>
              )}
   
              <div className="flex-1 overflow-auto">
                {!isLoading && isExtracted && similarImages.length == 0 ? (
                  <p className="text-center text-blue-600 font-bold text-lg">
                    No images found in the selected PDF. Please try another PDF.
                  </p>
                ) : similarImages.length > 0 && !isLoading ? (
                  <div className="space-y-4">
                    {similarImages.map((image, index) => (
                      <Card
                        key={index}
                        className="border border-gray-200 rounded-lg"
                      >
                        <CardContent className="flex items-center p-4">
                          <div className="w-32 h-32 flex-shrink-0">
                            <img
                              alt={image.image_name}
                              className="w-full h-full object-contain"
                              onError={(e) => {
                                e.target.src = "/placeholder.svg";
                              }}
                            />
                          </div>
                          <div className="ml-4 text-sm text-gray-600 space-y-2">
                            <p>
                              <strong>Similarity: </strong>
                              {(image.similarity * 100).toFixed(2)}%
                            </p>
                            <p>
                              <strong>Image name: </strong>
                              {image.image_name}
                            </p>
                            <p>
                              <strong>Title: </strong>
                              {image.title}
                            </p>
                            <p>
                              <strong>Caption: </strong>
                              {image.caption}
                            </p>
                            <p>
                              <strong>Authors: </strong>
                              {image.authors}
                            </p>
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
                ) : null}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
      {extractedImages.length > 0 && (
        <div className="flex flex-row items-end mx-6  gap-8">
          <div className="flex flex-row w-full gap-2   md:w-1/3 ">
            <div className="w-full md:w-3/5  ">
              <label className="block mb-1 font-medium">Select model</label>
              <select
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                className="w-full appearance-none py-2 px-3 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white pr-6"
              >
                {/* <option value="convnext_v2">ConvNeXt V2</option> */}
                {/* <option value="alexnet">AlexNet</option> */}
                {/* <option value="vgg16">VGG16</option> */}
                {/* <option value="InceptionV3">Inception V3</option> */}
                {/* <option value="InceptionV4">Inception V4</option> */}
                {/* <option value="InceptionResNetV2">Inception-ResNet V2</option> */}
                {/* <option value="MobileNetV2">MobileNetV2</option> */}
                <option value="ResNet101">ResNet101</option>
                <option value="ResNet101_Faiss">ResNet101 Faiss</option>
              </select>
              <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700 pt-6">
                <svg
                  className="h-4 w-4"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
            </div>
            <div className="w-full md:w-2/5 ">
              <label className="block mb-1 font-medium">
                Threshold (0 - 1)
              </label>
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
          <div className=" flex items-end cursor-pointer md:w-2/3 ">
            <Button
              variant="sky"
              onClick={handleSearchSimilar}
              className="bg-sky-400 w-full cursor-pointer h-11 hover:bg-sky-600 text-white rounded  text-lg shadow-md transition duration-200"
            >
              Find Similar Image
            </Button>
          </div>
        </div>
      )}

      <div className="px-6 mt-5 w-full">
        {extractedImages.length == 0 && sourcePdf != null && (
          <div className="">
            <Button
              onClick={handleExtractImages}
              disabled={!sourcePdf || isLoading}
              variant="sky"
              className="w-full h-11 text-white"
            >
              Submit
            </Button>
          </div>
        )}
      </div>

      {showPopup && sourcePdf && selectedImage && (
        <PopupDetails
          originalImage={{
            image_data: null,
            name: sourcePdf.name,
            caption:
              similarImages.find(
                (img) => img.image_id === selectedImage.image_id
              )?.caption || "N/A",
            doi: similarImages.find(
              (img) => img.image_id === selectedImage.image_id
            )?.doi,
          }}
          similarImage={selectedImage}
          onClose={() => setShowPopup(false)}
        />
      )}
      {isLoading && chosenImages.length > 0 && (
        <div
          className="fixed inset-0 z-50 flex flex-col justify-center items-center bg-white"
          style={{ backdropFilter: "blur(2px)" }}
        >
          <Spin size="large" />
          <p className="mt-4 text-lg font-medium text-sky-500 drop-shadow-sm">
            Please wait while the system is processing the images...
          </p>
        </div>
      )}
    </div>
  );
}

export default PdfSimilarityPage;
