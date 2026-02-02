import React from 'react';
import './ModelToggle.css';

const MODELS = [
  { id: 'fer', label: 'FER', icon: '🎯' },
  { id: 'hsemotion', label: 'HSE', icon: '🧠' },
  { id: 'deepface', label: 'Deep', icon: '🔬' },
];

function ModelToggle({ model, onModelChange }) {
  return (
    <div className="model-toggle">
      <span className="model-toggle-label">Model:</span>
      <div className="model-buttons">
        {MODELS.map((m) => (
          <button
            key={m.id}
            className={`model-btn ${model === m.id ? 'active' : ''}`}
            onClick={() => onModelChange(m.id)}
            title={m.id.toUpperCase()}
          >
            <span className="model-icon">{m.icon}</span>
            <span className="model-name">{m.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

export default ModelToggle;
