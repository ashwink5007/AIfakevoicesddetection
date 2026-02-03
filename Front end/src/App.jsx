import React, { useState, useEffect } from 'react';
import AudioUpload from './components/AudioUpload';
import AudioHistory from './components/AudioHistory';
import ResultDisplay from './components/ResultDisplay';
import './App.css';

export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [currentAudio, setCurrentAudio] = useState(null);

  const BACKEND_URL = 'http://localhost:5000/api/audio';

  // Load history from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('voiceshield_history');
    if (saved) {
      try {
        setHistory(JSON.parse(saved));
      } catch (e) {
        console.error('Error loading history:', e);
      }
    }
  }, []);

  // Save history to localStorage
  useEffect(() => {
    localStorage.setItem('voiceshield_history', JSON.stringify(history.slice(0, 50)));
  }, [history]);

  const handleFileSelect = (file) => {
    if (!file) return;

    setSelectedFile(file);
    setCurrentAudio({
      url: URL.createObjectURL(file),
      name: file.name,
      size: file.size
    });
    setError(null);
    setResult(null);
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please select an audio file');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${BACKEND_URL}/predict`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed');
      }

      // Add to history
      const newEntry = {
        id: Date.now(),
        fileName: data.fileName,
        prediction: data.prediction,
        confidence: data.confidence,
        uploadedAt: data.uploadedAt,
        audioUrl: currentAudio.url
      };

      setHistory([newEntry, ...history]);
      setResult(newEntry);
      setSelectedFile(null);
      setCurrentAudio(null);

    } catch (err) {
      console.error('Error:', err);
      setError(err.message || 'Failed to analyze audio. Make sure backend is running on port 5000.');
    } finally {
      setLoading(false);
    }
  };

  const handleHistorySelect = (entry) => {
    setResult(entry);
    setCurrentAudio({
      url: entry.audioUrl,
      name: entry.fileName
    });
  };

  const handleClearHistory = () => {
    if (window.confirm('Clear all history?')) {
      setHistory([]);
      setResult(null);
    }
  };

  return (
    <div className="app">
      <div className="app-container">
        {/* Main Content */}
        <main className="main-content">
          <div className="header">
            <h1>🎤 VoiceShield</h1>
            <p>AI-Powered Voice Authenticity Detection</p>
          </div>

          {/* Upload Section */}
          <section className="upload-section">
            <AudioUpload 
              onFileSelect={handleFileSelect}
              currentFile={selectedFile}
              loading={loading}
            />
          </section>

          {/* Current Audio Preview */}
          {currentAudio && (
            <section className="preview-section">
              <h3>📁 Selected File</h3>
              <div className="audio-preview">
                <div className="preview-info">
                  <p className="file-name">{currentAudio.name}</p>
                  {selectedFile && (
                    <p className="file-size">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  )}
                </div>
                <audio controls className="audio-player">
                  <source src={currentAudio.url} />
                  Your browser does not support the audio element.
                </audio>
              </div>
            </section>
          )}

          {/* Predict Button */}
          {selectedFile && (
            <button 
              className="predict-button"
              onClick={handlePredict}
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Analyzing...
                </>
              ) : (
                '✨ Check Voice'
              )}
            </button>
          )}

          {/* Error Display */}
          {error && (
            <div className="error-box">
              <span>⚠️</span>
              <p>{error}</p>
            </div>
          )}

          {/* Result Display */}
          {result && (
            <ResultDisplay result={result} />
          )}
        </main>

        {/* Sidebar - History */}
        <aside className="sidebar">
          <AudioHistory 
            history={history}
            onSelectEntry={handleHistorySelect}
            onClearHistory={handleClearHistory}
            selectedId={result?.id}
          />
        </aside>
      </div>
    </div>
  );
}
