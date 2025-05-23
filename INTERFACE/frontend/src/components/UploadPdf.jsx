import React, { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Card, CardContent } from "../components/Card";
import { CiBoxList } from "react-icons/ci";
import { CiTrash } from "react-icons/ci";
import { FaThList } from "react-icons/fa";
const UploadPdf = ({
  extractImages,
  onCloseDiv,
  onClearPdf,
  pdfUrl,
  sourcePdf,
  onSelectPdf,
}) => {
  const handleRemovePdf = () => {
    if (onClearPdf) {
      onClearPdf();
    }
  };
  const handleSourcePdfSelect = (e) => {
    if (onSelectPdf) {
      onSelectPdf(e);
    }
  };
  const handleClosePdfDiv = () => {
    if (onCloseDiv) {
      onCloseDiv();
    }
  };
  return (
    <>
      <CardContent className="">
        <div className="flex items-center bg-gray-50 py-2 rounded-md w-full">
          <h2 className="text-lg font-medium">Upload PDF</h2>

          {sourcePdf && (
            <button
              onClick={handleRemovePdf}
              title="Xóa PDF"
              className="p-2 rounded-full hover:bg-red-100 transition ml-auto m-0"
            >
              <CiTrash className="text-red-500 w-6 h-6 p-0 m-0" />
            </button>
          )}
          {extractImages.length > 0 && (
            <div
              className="p-2 rounded-full hover:bg-sky-100 transition flex justify-end "
              onClick={handleClosePdfDiv}
            >
              <CiBoxList className="w-6 h-6 text-sky-500" />
            </div>
          )}
        </div>

        <div className="flex flex-col gap-2">
          <div className="relative ">
            {sourcePdf && pdfUrl ? (
              <div
                className="relative overflow-auto"
                style={{
                  height: extractImages.length > 0 ? "250px" : "400px",
                }}
              >
                {" "}
                <iframe
                  src={pdfUrl}
                  width="100%"
                  height="100%"
                  title="PDF Preview"
                  style={{ border: "none" }}
                />
              </div>
            ) : (
              <div className="h-[400px] flex flex-col items-center justify-center ">
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
                  <p className="text-gray-500 text-center">
                    Drag and drop PDF here or click to select
                  </p>
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
    </>
  );
};

export default UploadPdf;
