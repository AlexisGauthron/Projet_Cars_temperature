import React, { useState, useEffect } from 'react';
import './DatasetCreator.css';

const EMOTIONS = [
  { id: 'angry', label: 'Angry', icon: '😠', color: '#e53935' },
  { id: 'disgust', label: 'Disgust', icon: '🤢', color: '#7cb342' },
  { id: 'fear', label: 'Fear', icon: '😨', color: '#8e24aa' },
  { id: 'happy', label: 'Happy', icon: '😊', color: '#fdd835' },
  { id: 'sad', label: 'Sad', icon: '😢', color: '#1e88e5' },
  { id: 'surprise', label: 'Surprise', icon: '😲', color: '#fb8c00' },
  { id: 'neutral', label: 'Neutral', icon: '😐', color: '#78909c' },
];

function DatasetCreator({ videoRef, onClose }) {
  const [counts, setCounts] = useState({});
  const [lastCapture, setLastCapture] = useState(null);
  const [flash, setFlash] = useState(false);
  const [datasetName, setDatasetName] = useState('my_dataset');

  // Charger les compteurs au démarrage
  useEffect(() => {
    fetchCounts();
  }, [datasetName]);

  const fetchCounts = async () => {
    try {
      const response = await fetch(`http://localhost:8000/dataset/counts?name=${datasetName}`);
      const data = await response.json();
      setCounts(data.counts || {});
    } catch (err) {
      console.error('Error fetching counts:', err);
    }
  };

  const captureEmotion = async (emotion) => {
    if (!videoRef.current) return;

    // Créer un canvas pour capturer la frame
    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    // Convertir en base64
    const imageData = canvas.toDataURL('image/jpeg', 0.9);

    // Flash effect
    setFlash(true);
    setTimeout(() => setFlash(false), 200);

    try {
      const response = await fetch('http://localhost:8000/dataset/capture', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: imageData,
          emotion: emotion,
          dataset_name: datasetName
        })
      });

      const result = await response.json();

      if (result.success) {
        // Mettre à jour le compteur
        setCounts(prev => ({
          ...prev,
          [emotion]: (prev[emotion] || 0) + 1
        }));
        setLastCapture({ emotion, timestamp: Date.now() });
      }
    } catch (err) {
      console.error('Error capturing:', err);
    }
  };

  const totalImages = Object.values(counts).reduce((a, b) => a + b, 0);

  return (
    <div className="dataset-creator-overlay">
      <div className="dataset-creator">
        {/* Flash effect */}
        {flash && <div className="capture-flash" />}

        {/* Header */}
        <div className="dataset-header">
          <h2>Dataset Creator</h2>
          <div className="dataset-name-input">
            <label>Dataset:</label>
            <input
              type="text"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value.replace(/[^a-zA-Z0-9_-]/g, ''))}
              placeholder="my_dataset"
            />
          </div>
          <button className="close-btn" onClick={onClose}>✕</button>
        </div>

        {/* Stats */}
        <div className="dataset-stats">
          <span className="total-count">Total: {totalImages} images</span>
          {lastCapture && (
            <span className="last-capture">
              Dernière: {EMOTIONS.find(e => e.id === lastCapture.emotion)?.icon} {lastCapture.emotion}
            </span>
          )}
        </div>

        {/* Emotion buttons grid */}
        <div className="emotion-buttons-grid">
          {EMOTIONS.map((emotion) => (
            <button
              key={emotion.id}
              className="emotion-capture-btn"
              style={{ '--emotion-color': emotion.color }}
              onClick={() => captureEmotion(emotion.id)}
            >
              <span className="emotion-icon">{emotion.icon}</span>
              <span className="emotion-label">{emotion.label}</span>
              <span className="emotion-count">{counts[emotion.id] || 0}</span>
            </button>
          ))}
        </div>

        {/* Instructions */}
        <div className="dataset-instructions">
          <p>Cliquez sur une émotion pour capturer votre visage avec cette expression.</p>
          <p className="dataset-path">📁 backend/data/{datasetName}/</p>
        </div>
      </div>
    </div>
  );
}

export default DatasetCreator;
