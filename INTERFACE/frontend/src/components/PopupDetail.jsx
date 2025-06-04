import React from "react";
import { ImageIcon } from "lucide-react";

const PopupDetails = ({ originalImage, similarImage, onClose, type }) => {
  console.log(originalImage, similarImage);
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm ">
      <div className="absolute inset-0" onClick={onClose}></div>

      <div className="relative h-[600px] overflow-auto z-50 bg-white rounded-xl shadow-2xl p-8 max-w-6xl w-full mx-4 md:mx-8 grid grid-cols-1 md:grid-cols-2 gap-8 animate-fade-in">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-red-600 text-3xl font-bold transition cursor-pointer"
          aria-label="Close"
        >
          x
        </button>

        <div className="flex flex-col gap-4">
          <h2 className="text-xl font-bold text-blue-700 flex items-center gap-2 ">
            <ImageIcon className="w-5 h-5 text-blue-500" />
            Original Image
          </h2>
          <div className="flex items-center justify-center  h-[300px] rounded-lg border border-gray-200 shadow-md ">
            <img
              src={`data:image/png;base64,${originalImage?.base64}`}
              alt={originalImage.name}
              className=" max-h-[300px] object-contain "
            />
          </div>
          <div className="text-sm text-gray-700 space-y-1 mt-3">
            <p className="text-lg font-semibold text-sky-500 text-center">
              Predicted class: {originalImage.predicted_class} – Confidence:{" "}
              {originalImage.confidence?.toFixed(2) || "N/A"}%
            </p>
            {type == "pdf" && (
              <>
                <p>
                  <strong>Title:</strong> {originalImage.title}
                </p>
                <p>
                  <strong>Authors:</strong> {originalImage.authors}
                </p>
                <p>
                  <strong>Accepted Date:</strong> {originalImage?.accepted_date}
                </p>
                {originalImage.doi && (
                  <p>
                    <a
                      href={`https://doi.org/${originalImage.doi}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:underline break-words"
                    >
                      🔗 DOI: {originalImage.doi}
                    </a>
                  </p>
                )}
              </>
            )}
          </div>
        </div>

        {/* Similar Image Section */}
        <div className="flex flex-col gap-4">
          <h2 className="text-xl font-bold text-green-700 flex items-center gap-2">
            <ImageIcon className="w-5 h-5 text-green-500" />
            Similar Image
          </h2>
          <div className="flex items-center justify-center   h-[300px] rounded-lg border border-gray-200 shadow-md ">
            {type == "pdf" ? (
              <img
                src={`http://127.0.0.1:5001/dataset/${originalImage?.predicted_class}/${similarImage?.image_field_name}`}
                alt={similarImage.image_name}
                className=" max-h-[300px] object-contain "
              />
            ) : (
              <img
                src={`http://127.0.0.1:5001/dataset/${originalImage?.predicted_class}/${similarImage?.image_field_name}`}
                alt={similarImage.image_name}
                className=" max-h-[300px] object-contain "
              />
            )}
          </div>
          <div className="text-sm text-gray-700 space-y-1 mt-2">
            <p className="text-lg font-semibold text-sky-500 text-center">
              Similarity: {similarImage.similarity?.toFixed(2)}%
            </p>
            <p>
              <strong>Title:</strong> {similarImage.title || "N/A"}
            </p>
            <p>
              <strong>Authors:</strong> {similarImage.authors || "N/A"}
            </p>
            <p>
              <strong>Caption:</strong> {similarImage.caption || "N/A"}
            </p>
            <p>
              <strong>Accepted Date:</strong>{" "}
              {similarImage.accepted_date}
            </p>
            {similarImage.doi && (
              <p>
                <a
                  href={`https://doi.org/${similarImage.doi}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-green-600 hover:underline break-words"
                >
                  🔗 DOI: {similarImage.doi}
                </a>
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PopupDetails;
