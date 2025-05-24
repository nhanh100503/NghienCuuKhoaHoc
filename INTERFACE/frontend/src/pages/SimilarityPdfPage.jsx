import React, { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import Button from "../components/Button";
import PopupDetails from "../components/PopupDetail";
const SimilarityPdfPage = () => {
  const location = useLocation();
  const { results, class_counts, total } = location.state || {};
  const [isShowPopup, setIsShowPopup] = useState(false);
  const [originalImage, setOriginImage] = useState(null);
  const [simImage, setSimImage] = useState(null);
  const [selectedClass, setSelectedClass] = useState("All classes");
  const classes = [
    "All classes",
    ...new Set(results?.map((img) => img.predicted_class)),
  ];
  const [isCloseDiv, setIsCloseDiv] = useState(false);
  const modelMap = {
    vgg16: "VGG16",
    alexnet: "AlexNet",
    convnext_v2: "ConvNeXt V2",
    ResNet101: "ResNet101",
    EfficentNetB0: "EfficentNetB0",
    ResNet101: "ResNet101",
    MobileNetV2: "MobileNetV2",
    InceptionResNetV2: "Inception-ResNet V2",
    InceptionV3: "Inception V3",
    InceptionV4: "Inception V4",
  };
  console.log(results);
  const handleShowPopup = (image, sim) => {
    setIsShowPopup(true);
    setOriginImage(image);
    setSimImage(sim);
  };
  const filteredResults =
    selectedClass === "All classes"
      ? results
      : results.filter((img) => img.predicted_class === selectedClass);
  return (
    <div className=" m-6">
      <div className="flex items-center relative">
        <Link to="/pdf-similarity" >
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
            Back
          </Button>
        </Link>
        <h1 className="text-3xl font-extrabold absolute left-1/2 transform -translate-x-1/2 text-sky-500 drop-shadow-md">
          Similar Images Result
        </h1>
      </div>
      <div className="flex items-center justify-between mt-10 pl-4">
        <div>
          <p className="text-lg font-semibold text-sky-500 text-center">
            Model: {modelMap[filteredResults[0].model] || "Unknown"} –
            Threshold: {filteredResults[0].threshold?.toFixed(2) || "N/A"}  
          </p>
        </div>
        <div className="pb-2 pr-4 flex flex-wrap gap-3">
          {classes.map((cls) => (
            <button
              key={cls}
              onClick={() => setSelectedClass(cls)}
              className={`px-5 py-2 rounded-full text-sm font-medium shadow-sm transition-all duration-200 cursor-pointer
          ${
            selectedClass === cls
              ? "bg-sky-500 text-white hover:bg-blue-500"
              : "bg-gray-100 text-gray-700 hover:bg-gray-200"
          }`}
            >
              {cls}
            </button>
          ))}
        </div>
      </div>

      {filteredResults?.map((image, index) => (
        <div
          key={index}
          className=" rounded-xl shadow p-4 mb-8 mt-0 bg-white m-4 border border-gray-200"
        >
          <div className="grid grid-cols-1 md:grid-cols-[300px_1fr] gap-8">
            {/* Ảnh nguồn */}
            <div className="flex flex-col  justify-center">
              <img
                className="rounded object-cover w-[300px] h-auto"
                src={`data:image/png;base64,${image?.base64}`}
                alt="Source"
              />
              <div className="mt-4 ">
                <p className="text-center text-lg font-semibold text-sky-700 mb-2">
                  Input Image
                </p>
                <p className="text-sm text-gray-700 mb-1">
                  <span className="font-medium">Image</span> {index + 1}
                </p>
                <p className="text-sm text-gray-700">
                  <span className="font-medium">Predicted class:</span>{" "}
                  {image.predicted_class}
                </p>
                <p className="text-sm text-gray-700">
                  <span className="font-medium">Confidence:</span>{" "}
                  {image?.confidence.toFixed(2)}%
                </p>
              </div>
            </div>

            {/* Ảnh tương tự */}
            <div className="text-start">
              <p className="font-semibold text-gray-700 mb-2">
                Similar Images (Total: {image?.similar_images.length})
              </p>
              <div className="grid grid-cols-3 gap-2 max-h-[400px] overflow-y-auto pr-2">
                {image?.similar_images?.map((sim, simIndex) => (
                  <>
                    <div
                      className="border border-gray-300 rounded-lg pt-3 pb-2 shadow-sm m-1"
                      key={index}
                    >
                      <div
                        key={simIndex}
                        className="flex flex-col items-center  "
                      >
                        <img
                          width={350}
                          height={200}
                          src={
                            sim?.image_data.startsWith("data:image")
                              ? sim.image_data
                              : `data:image/png;base64,${sim.image_data}`
                          }
                          alt={`Similar ${simIndex}`}
                          className=" object-cover rounded w-[300px] h-[200px]"
                        />
                      </div>
                      <div className="mt-2 flex flex-col text-sm text-gray-700 space-y-1 px-2 h-[150px] overflow-auto">
                        <p>
                          <span className="font-semibold text-gray-800">
                            Confidence:
                          </span>{" "}
                          {sim.similarity.toFixed(2)}%
                        </p>
                        <p>
                          <span className="font-semibold text-gray-800">
                            Caption:
                          </span>{" "}
                          {sim.caption}
                        </p>
                        <p>
                          <span className="font-semibold text-gray-800">
                            DOI:
                          </span>{" "}
                          {sim.doi}
                        </p>
                        <p
                          className="text-end mt-auto text-blue-500 cursor-pointer hover:text-blue-700 hover:font-semibold transition-colors duration-200"
                          onClick={() => handleShowPopup(image, sim)}
                        >
                          View Detail
                        </p>
                      </div>
                    </div>
                  </>
                ))}
              </div>
              {/* <div className="p-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-[500px] overflow-auto scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-gray-100 pr-2">
                  {image?.similar_images?.map((sim, index) => (
                    <div
                      key={index}
                      className="flex flex-col md:flex-row items-center bg-white border border-gray-200 rounded-xl shadow-md overflow-hidden transition hover:shadow-xl"
                    >
                      <div className="w-full md:w-1/2 h-full p-3 flex justify-center bg-gray-50">
                        <img
                          src={sim.image_data || "/placeholder.svg"}
                          alt={`Similar ${index}`}
                          className="max-h-[200px] object-contain rounded-md"
                        />
                      </div>

                      <div className="w-full md:w-1/2 p-4 text-sm text-gray-700 space-y-1">
                        <p>
                          <strong>Title:</strong>{" "}
                            {sim.title || "N/A"}
                        </p>
                        <p>
                          <strong>Authors:</strong>{" "}
                            {sim.authors || "N/A"}
                        </p>
                        <p>
                          <strong>Caption:</strong>{" "}
                            {sim.caption || "N/A"}
                        </p>
                        <p>
                          <strong>DOI:</strong>{" "}
                          {sim.doi ? (
                            <a
                              href={`https://doi.org/${sim.doi}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-600 hover:underline break-all"
                            >
                              {sim.doi}
                            </a>
                          ) : (
                            "N/A"
                          )}
                        </p>
                        <p className="text-end text-blue-500 cursor-pointer" onClick={() => handleShowPopup(image, sim)}>View detail</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div> */}
            </div>
          </div>
        </div>
      ))}
      {isShowPopup && (
        <PopupDetails
          originalImage={originalImage}
          similarImage={simImage}
          onClose={() => setIsShowPopup(false)}
        ></PopupDetails>
      )}
    </div>
  );
};

export default SimilarityPdfPage;
