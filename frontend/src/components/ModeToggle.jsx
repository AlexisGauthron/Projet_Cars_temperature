import React from 'react';
import './ModeToggle.css';

function ModeToggle({ mode, onModeChange }) {
  return (
    <div className="mode-toggle">
      <button
        className={`mode-btn ${mode === 'single' ? 'active' : ''}`}
        onClick={() => onModeChange('single')}
      >
        <span className="mode-icon">1</span>
        <span className="mode-label">Solo</span>
      </button>
      <button
        className={`mode-btn ${mode === 'multi' ? 'active' : ''}`}
        onClick={() => onModeChange('multi')}
      >
        <span className="mode-icon">+</span>
        <span className="mode-label">Multi</span>
      </button>
    </div>
  );
}

export default ModeToggle;
