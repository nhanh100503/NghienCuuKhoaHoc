"use client"

import { useState } from "react"
import { Link } from "react-router-dom"
import { Card, CardContent } from "../components/Card"
import Button from "../components/Button"
import ImageDropZone from "../components/ImageDropZone"

function ClassificationPage() {
    const [selectedImage, setSelectedImage] = useState(null)
    const [selectedModel, setSelectedModel] = useState("")
    const [classificationResult, setClassificationResult] = useState(null)
    const [error, setError] = useState(null)
    const [isLoading, setIsLoading] = useState(false)

    const handleImageSelect = (imageData) => {
        setSelectedImage(imageData)
        setClassificationResult(null)
        setError(null)
        setIsLoading(false)
    }

    const clearImage = () => {
        setSelectedImage(null)
        setClassificationResult(null)
        setError(null)
        setIsLoading(false)
    }

    const handleModelChange = (event) => {
        setSelectedModel(event.target.value)
        setClassificationResult(null)
        setError(null)
        setIsLoading(false)
    }

    const handleSubmit = async () => {
        setIsLoading(true)
        setClassificationResult(null)
        setError(null)

        const formData = new FormData()
        formData.append('image', selectedImage.file)
        formData.append('model', selectedModel)

        try {
            const response = await fetch('http://localhost:5000/classify', {
                method: 'POST',
                body: formData
            })
            const result = await response.json()
            if (response.ok) {
                setClassificationResult(result)
                setError(null)
            } else {
                setError(result.error || 'Failed to classify image')
                setClassificationResult(null)
            }
        } catch (err) {
            setError('Failed to connect to the server')
            setClassificationResult(null)
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <div className="flex flex-col min-h-screen p-6">
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
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                        </svg>
                        Back to home
                    </Button>
                </Link>
            </div>

            <h1 className="text-2xl font-bold mb-8 text-center">Image Classification</h1>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Left Column - Image Upload */}
                <div className="flex flex-col gap-4">
                    <Card className="h-[400px] cursor-pointer bg-white border border-gray-200 rounded-lg shadow-sm">
                        <CardContent>
                            <div className="flex items-center justify-between mb-4">
                                <h2 className="text-lg font-medium">Upload image</h2>
                                {selectedImage && (
                                    <button
                                        onClick={clearImage}
                                        className="flex items-center gap-1 bg-red-500 hover:bg-red-600 text-white rounded-md px-3 py-1.5 text-sm transition-all duration-200 shadow-sm hover:shadow-md active:bg-red-700"
                                        title="Xóa ảnh"
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
                                    {selectedImage ? (
                                        <div className="relative h-full">
                                            <img
                                                src={selectedImage.dataUrl || "/placeholder.svg"}
                                                alt="Selected"
                                                className="max-h-full max-w-full object-contain mx-auto"
                                            />
                                        </div>
                                    ) : (
                                        <ImageDropZone
                                            onImageSelect={handleImageSelect}
                                            className="h-full flex flex-col items-center justify-center"
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
                                            <p className="text-gray-500 text-center">Drag and drop image here or click to select</p>
                                        </ImageDropZone>
                                    )}
                                </div>

                                {selectedImage ? (
                                    <p className="text-sm text-gray-700">
                                        <strong>Selected: </strong> {selectedImage.file.name}
                                    </p>
                                ) : (
                                    <p className="text-sm text-gray-700">No image selected</p>
                                )}
                            </div>
                        </CardContent>
                    </Card>
                </div>

                {/* Right Column - Classification Results */}
                <div className="flex flex-col gap-4">
                    <Card className="h-[400px] bg-white border border-gray-200 rounded-lg shadow-sm">
                        <CardContent className="h-full flex flex-col">
                            <h2 className="text-lg font-semibold text-gray-800 mb-4">Classification results</h2>
                            <div className="flex-1 flex flex-col items-center justify-center">
                                {isLoading ? (
                                    <div className="text-center">
                                        <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                                        <p className="text-sm text-gray-600">Classifying...</p>
                                    </div>
                                ) : classificationResult ? (
                                    <div className="w-full">
                                        <p className="text-sm text-gray-600 mb-4 text-center">
                                            <strong>Model Being Used:</strong> {
                                                classificationResult.model === "model1" ? "ResNet101" :
                                                    classificationResult.model === "model2" ? "ResNet101 tăng cường" :
                                                        classificationResult.model === "model3" ? "MobileNetV2" :
                                                            classificationResult.model === "model4" ? "MobileNetV2 tăng cường" :
                                                                classificationResult.model === "model5" ? "EfficientNetB2" :
                                                                    classificationResult.model === "model6" ? "EfficientNetB2 tăng cường" :
                                                                        classificationResult.model
                                            }
                                        </p>
                                        <p className="text-sm text-gray-600 mb-2 text-center"><strong>TOP 5 PREDICTIONS</strong></p>
                                        <div className="space-y-2">
                                            {classificationResult.predictions.map((pred, index) => (
                                                <div
                                                    key={index}
                                                    className={`flex justify-between items-center px-4 py-2 rounded-md ${index === 0 ? 'bg-blue-100' : ''}`}
                                                >
                                                    <span className={`${index === 0 ? 'text-blue-600 font-extrabold' : 'font-semibold'}`}>{index + 1}</span>
                                                    <span className={`text-sm ${index === 0 ? 'text-blue-600 font-extrabold' : 'font-semibold'}`}>{pred.class}</span>
                                                    <span className={`text-sm ${index === 0 ? 'text-blue-600 font-extrabold' : 'text-gray-700'}`}>
                                                        {(pred.confidence * 100).toFixed(2)}%
                                                    </span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                ) : error ? (
                                    <div className="text-center">
                                        <p className="text-sm text-red-600">{error}</p>
                                    </div>
                                ) : (
                                    <div className="text-center">
                                        <svg
                                            xmlns="http://www.w3.org/2000/svg"
                                            className="h-12 w-12 mx-auto mb-4 text-gray-400"
                                            fill="none"
                                            viewBox="0 0 24 24"
                                            stroke="currentColor"
                                            strokeWidth={1.5}
                                        >
                                            <path
                                                strokeLinecap="round"
                                                strokeLinejoin="round"
                                                d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                                            />
                                        </svg>
                                        <p className="text-base text-gray-500">Classification results</p>
                                    </div>
                                )}
                            </div>
                        </CardContent>
                    </Card>
                </div>
            </div>
            <div className="flex justify-center mt-6">
                <div className="relative w-full md:w-1/2">
                    <select
                        value={selectedModel}
                        onChange={handleModelChange}
                        className="w-full appearance-none py-2 px-3 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white pr-8"
                    >
                        <option value="">Select model</option>
                        <option value="model1">ResNet101</option>
                        <option value="model2">Augmented ResNet101</option>
                        <option value="model3">MobileNetV2</option>
                        <option value="model4">Augmented MobileNetV2</option>
                        <option value="model5">EfficientNetB0</option>
                        <option value="model6">Augmented EfficientNetB0</option>
                    </select>
                    <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
                        <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                        </svg>
                    </div>
                </div>
            </div>
            {/* Submit Button - Improved */}
            <div className="flex justify-center mt-6">
                <button
                    onClick={handleSubmit}
                    className="w-full md:w-1/3 bg-blue-500 hover:bg-blue-600 text-white rounded-md py-2 px-4 font-medium transition-all duration-200 shadow-sm hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
                    disabled={!selectedImage || !selectedModel || isLoading}
                >
                    {isLoading ? (
                        <span className="flex items-center justify-center gap-2">
                            <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            Processing...
                        </span>
                    ) : 'Classification'}
                </button>
            </div>
        </div>
    )
}

export default ClassificationPage


