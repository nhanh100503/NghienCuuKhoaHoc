import React, { useState } from "react";
import { Link } from "react-router-dom";
import { Card, CardContent } from "../components/Card";
import Button from "../components/Button";
import ImageDropZone from "../components/ImageDropZone";
import PopupDetails from "../components/PopupDetail";
import axios from "axios";
import { CiImageOn } from "react-icons/ci";
import { CiTrash } from "react-icons/ci";
import { FaRegFilePdf } from "react-icons/fa6";

function SimilarityPage() {
  const [sourceImage, setSourceImage] = useState(null);
  const [similarImages, setSimilarImages] = useState([]);
  const [predictedClass, setPredictedClass] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelName, setModelName] = useState("convnext_v2");
  const [threshold, setThreshold] = useState(0.8);
  const [showPopup, setShowPopup] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [totalImages, setTotalImages] = useState(0);

  const handleSourceImageSelect = (imageData) => {
    setSourceImage(imageData);
    setSimilarImages([]);
    setPredictedClass(null);
    setError(null);
  };

  const encodeImageToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = () => {
        resolve(reader.result); 
      };

      reader.onerror = (error) => {
        reject(error);
      };

      reader.readAsDataURL(file); 
    });
  };

  const handleSearchSimilar = async () => {
    setIsLoading(true);
    setError(null);
    const base64Image = await encodeImageToBase64(sourceImage.file);
    try {
      const response = await axios.post(
        "http://127.0.0.1:5003/similarity-image",
        {
          model_name: modelName,
          threshold: threshold,
          images: base64Image,
        },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      console.log(response);
      setSimilarImages(response?.data?.results);
    } catch (err) {
      setError(err.message || "Có lỗi xảy ra khi tìm kiếm ảnh tương tự");
    } finally {
      setIsLoading(false);
    }
  };

  const clearSourceImage = () => {
    setSourceImage(null);
    setSimilarImages([]);
    setTotalImages(0);
    setPredictedClass(null);
    setError(null);
  };

  const handleShowImageDetail = (originalImage, similarImage) => {
    setSelectedImage(similarImage); // Pass the clicked similar image
    setShowPopup(true);
    console.log("Selected image:", image);
  };

  return (
    <div className="flex flex-col h-screen p-6">
      <div className="mb-6 flex items-center justify-between relative">
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
            Single Image Similarity Search
          </h1>
        </div>
        <div className="">
          <Link to="/pdf-similarity">
            <FaRegFilePdf className="w-7 h-7 text-sky-500" />
          </Link>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-[1.5fr_3fr] gap-8 mb-8">
        <div className="flex flex-col gap-4">
          <Card className="cursor-pointer bg-white border border-gray-200 rounded-lg shadow-sm py-4">
            <CardContent>
              <div className="flex items-center justify-between mb-4 ">
                <h2 className="text-lg font-medium">Upload image</h2>
                {sourceImage && (
                  <button
                    onClick={clearSourceImage}
                    className="p-2 rounded-full hover:bg-red-100 transition ml-auto m-0 cursor-pointer"
                  >
                    <CiTrash className="text-red-500 w-6 h-6 p-0 m-0" />
                  </button>
                )}
              </div>
              <div className="flex flex-col gap-4">
                <div className=" h-[345px] flex items-center justify-center">
                  {sourceImage ? (
                    <div className="">
                      <img
                        src={sourceImage.dataUrl }
                        className=" max-h-[350px] max-w-[450px] object-contain mx-auto"
                      />
                    </div>
                  ) : (
                    <ImageDropZone
                      onImageSelect={handleSourceImageSelect}
                      className=" flex flex-col items-center justify-center "
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
                      <p className="text-gray-500 text-center">
                        Drag and drop image here or click to select
                      </p>
                    </ImageDropZone>
                  )}
                </div>
                {sourceImage ? (
                  <p className="text-sm text-gray-700">
                    <strong className="">Selected: </strong> {sourceImage.file.name}
                  </p>
                ) : (
                  <p className="text-sm text-gray-700"><strong>No image selected</strong> </p>
                )}
              </div>
            </CardContent>
          </Card>
          <div className="flex flex-col md:flex-row gap-4 justify-center items-center mb-4">
            <div className="relative w-full md:w-3/5">
              <label className="block mb-1 font-medium">Select model</label>
              <select
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                className="w-full appearance-none py-2 px-3 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white pr-8"
              >
                <option value="convnext_v2">ConvNeXt V2</option>
                {/* <option value="alexnet">AlexNet</option> */}
                <option value="vgg16">VGG16</option>
                {/* <option value="InceptionV3">Inception V3</option> */}
                <option value="InceptionV4">Inception V4</option>
                {/* <option value="InceptionResNetV2">Inception ResNet</option> */}
                {/* <option value="MobileNetV2">MobileNetV2</option> */}
                <option value="ResNet101">ResNet101</option>
                {/* <option value="EfficientNetB0">EfficientNetB0</option> */}  
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
            <div className="relative w-full md:w-2/5">
              <label className="block mb-1 font-medium">
                Threshold (0 - 1)
              </label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="border border-gray-300 rounded px-3 py-2 w-full"
              />
            </div>
          </div>
          <Button
            onClick={handleSearchSimilar}
            disabled={!sourceImage || isLoading}
            variant="sky"
          >
            {isLoading ? "Processing ..." : "Submit"}
          </Button>

          {error && <p className="text-red-500 text-center">{error}</p>}
        </div>

        <Card className="bg-white border border-gray-200 rounded-lg shadow-sm h-[620px] overflow-hidden">
          <CardContent className="p-4 h-full flex flex-col">
            <h2 className="text-lg font-medium mb-4">Result</h2>
            {isLoading ? (
              <div className="text-center pt-50">
                <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                {/* <p className="text-sm text-gray-600">Classifying...</p> */}
              </div>
            ) : (
              similarImages?.predicted_class && (
                <div className="flex-1 overflow-auto ">
                  <div className="text-center mb-4">
                    <p className="text-gray-600">
                      Classification results:{" "}
                      <span className="font-bold">
                        {similarImages?.predicted_class}
                      </span>
                    </p>
                    <p className="text-gray-600">
                      Total similar images:{" "}
                      <span className="font-bold">
                        {similarImages?.total_similar_images}
                      </span>
                    </p>
                  </div>
                  <div>
                    {similarImages?.similar_images?.length > 0 ? (
                      <div className="">
                        <div className="space-y-4 ">
                          {similarImages?.similar_images?.map(
                            (image, index) => (
                              <div
                                key={index}
                                className="border border-gray-200 rounded-lg "
                              >
                                <CardContent className="flex items-center p-4  ">
                                  <div className=" flex items-center justify-center md:w-1/2">
                                    <img
                                      // src={image?.image_data}
                                      src={`http://127.0.0.1:5003/dataset/${similarImages?.predicted_class}/${image?.image_field_name}`}
                                      alt={image.image_name}
                                      className="w-[400px] h-[250px]"
                                    />
                                  </div>
                                  <div className="ml-4 text-sm text-gray-600 space-y-2 md:w-1/2">
                                    <p>
                                      <strong>Number: </strong>
                                      {index + 1}
                                    </p>
                                    <p>
                                      <strong>Similarity: </strong>
                                      {image.similarity.toFixed(2)}%
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
                                        className="text-end mt-auto text-blue-500 cursor-pointer hover:text-blue-700 hover:font-semibold transition-colors duration-200"
                                        onClick={() =>
                                          handleShowImageDetail(
                                            similarImages,
                                            image
                                          )
                                        }
                                      >
                                        View Detail
                                      </p>
                                    </div>
                                  </div>
                                </CardContent>
                              </div>
                            )
                          )}
                        </div>
                      </div>
                    ) : !isLoading && predictedClass ? (
                      <p className="text-center text-gray-600">
                        No similar images found with current threshold.
                      </p>
                    ) : null}
                  </div>
                </div>
              )
            )}
          </CardContent>
        </Card>
      </div>
      {showPopup && sourceImage && selectedImage && (
        <PopupDetails
          originalImage={similarImages}
          similarImage={selectedImage}
          onClose={() => setShowPopup(false)}
          type="image"
        />
      )}
    </div>
  );
}

export default SimilarityPage;
