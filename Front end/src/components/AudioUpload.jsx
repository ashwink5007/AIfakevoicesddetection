import React, { useRef } from 'react';

export default function AudioUpload({ onFileSelect, currentFile, loading }) {
  const inputRef = useRef(null);
  const dragRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    dragRef.current?.classList.add('drag-active');
  };

  const handleDragLeave = () => {
    dragRef.current?.classList.remove('drag-active');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    dragRef.current?.classList.remove('drag-active');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('audio/') || file.name.match(/\.(wav|mp3|flac|ogg|m4a|webm)$/i)) {
        onFileSelect(file);
      } else {
        alert('Please drop an audio file');
      }
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileSelect(file);
    }
  };

  return (
    <div
      ref={dragRef}
      className="upload-area"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="upload-content">
        <p className="upload-icon">📤</p>
        <h3>Drag & Drop Audio Here</h3>
        <p>or click to select</p>
        <input
          ref={inputRef}
          type="file"
          accept="audio/*"
          onChange={handleFileSelect}
          className="upload-input"
          disabled={loading}
        />
        <button
          className="select-button"
          onClick={() => inputRef.current?.click()}
          disabled={loading}
        >
          Choose File
        </button>
        <p className="upload-hint">
          Supported: WAV, MP3, FLAC, OGG, M4A, WebM (Max 50MB)
        </p>
      </div>
    </div>
  );
}
