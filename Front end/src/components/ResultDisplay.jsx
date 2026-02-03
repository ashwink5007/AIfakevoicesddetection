import React from 'react';

export default function ResultDisplay({ result }) {
  const isPredictionReal = result.prediction === 'REAL';

  return (
    <div className={`result-container ${isPredictionReal ? 'real' : 'fake'}`}>
      <div className="result-badge">
        {isPredictionReal ? '🟢' : '🔴'}
      </div>

      <div className="result-content">
        <h2 className="result-title">Prediction Result</h2>

        <div className={`prediction-badge ${isPredictionReal ? 'badge-real' : 'badge-fake'}`}>
          {result.prediction}
        </div>

        <div className="confidence-section">
          <label>Confidence Score</label>
          <div className="confidence-bar">
            <div
              className={`confidence-fill ${isPredictionReal ? 'fill-real' : 'fill-fake'}`}
              style={{ width: `${result.confidence}%` }}
            />
          </div>
          <p className="confidence-value">{result.confidence}%</p>
        </div>

        <div className="result-metadata">
          <div className="metadata-item">
            <span className="label">File</span>
            <span className="value truncate">{result.fileName}</span>
          </div>
          <div className="metadata-item">
            <span className="label">Analyzed</span>
            <span className="value">
              {new Date(result.uploadedAt).toLocaleString()}
            </span>
          </div>
        </div>

        <div className="result-description">
          {isPredictionReal ? (
            <p>✅ This voice appears to be <strong>REAL</strong> and authentic.</p>
          ) : (
            <p>⚠️ This voice may be <strong>AI-GENERATED</strong> or manipulated.</p>
          )}
        </div>
      </div>
    </div>
  );
}
