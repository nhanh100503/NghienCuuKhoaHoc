"use client"

import { useState, useRef } from "react"

function ImageDropZone({ onImageSelect, color = "emerald", className = "", children }) {
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef(null)

  const colorClass = color === "sky" ? "image-drop-area-blue" : ""

  const handleDragEnter = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (!isDragging) setIsDragging(true)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0]
      if (file.type.startsWith("image/")) {
        processFile(file)
      }
    }
  }

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0])
    }
  }

  const processFile = (file) => {
    const reader = new FileReader()
    reader.onload = () => {
      onImageSelect({
        file,
        dataUrl: reader.result,
      })
    }
    reader.readAsDataURL(file)
  }

  const handleClick = () => {
    fileInputRef.current.click()
  }

  return (
    <div
      className={`image-drop-area ${colorClass} ${isDragging ? "active" : ""} ${className}`}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      onClick={handleClick}
    >
      <input type="file" ref={fileInputRef} className="hidden" accept="image/*" onChange={handleFileSelect} />
      {children}
    </div>
  )
}

export default ImageDropZone
