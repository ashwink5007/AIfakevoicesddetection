import React from 'react';

export default function AudioHistory({ history, onSelectEntry, onClearHistory, selectedId }) {
  return (
    <div className="history-panel">
      <div className="history-header">
        <h2>📜 History</h2>
        {history.length > 0 && (
          <button
            className="clear-button"
            onClick={onClearHistory}
            title="Clear all history"
          >
            🗑️
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <div className="history-empty">
          <p>No analyses yet</p>
          <p className="hint">Upload and analyze audio files to see history</p>
        </div>
      ) : (
        <div className="history-list">
          {history.map((entry) => (
            <div
              key={entry.id}
              className={`history-item ${entry.prediction.toLowerCase()} ${
                selectedId === entry.id ? 'selected' : ''
              }`}
              onClick={() => onSelectEntry(entry)}
            >
              <div className="history-badge">
                {entry.prediction === 'REAL' ? '🟢' : '🔴'}
              </div>
              <div className="history-info">
                <p className="history-name" title={entry.fileName}>
                  {entry.fileName.length > 20
                    ? entry.fileName.substring(0, 17) + '...'
                    : entry.fileName}
                </p>
                <p className="history-confidence">
                  {entry.confidence}% confidence
                </p>
                <p className="history-time">
                  {new Date(entry.uploadedAt).toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit'
                  })}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
