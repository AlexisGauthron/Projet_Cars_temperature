import React from 'react';
import { TEMPERATURE_CONFIG } from '../config/temperature';
import './CameraView.css';

// Mapping des états thermiques PPG vers les labels et icônes
const PPG_THERMAL_STATES = {
  cold: { label: 'Froid', icon: '🥶', color: '#2196f3' },
  cool: { label: 'Frais', icon: '❄️', color: '#4fc3f7' },
  neutral: { label: 'Neutre', icon: '😊', color: '#4caf50' },
  warm: { label: 'Chaud', icon: '🌡️', color: '#ff9800' },
  hot: { label: 'Très chaud', icon: '🔥', color: '#f44336' },
  unknown: { label: 'Analyse...', icon: '⏳', color: '#9e9e9e' },
  error: { label: 'Erreur', icon: '⚠️', color: '#9e9e9e' },
};

const CameraView = ({ videoRef, canvasRef, annotatedImage, emotion, error, vlmQuestion, vlmOptions, onVLMResponse, temperature, primaryEmotion, ppgData }) => {
  // Afficher l'émotion du détecteur primary (FER): 'confortable' ou 'inconfortable'
  // Capitaliser la première lettre pour l'affichage
  const emotionLabel = primaryEmotion
    ? primaryEmotion.charAt(0).toUpperCase() + primaryEmotion.slice(1)
    : '';

  // Données PPG formatées
  const ppgState = ppgData ? PPG_THERMAL_STATES[ppgData.thermal_state] || PPG_THERMAL_STATES.unknown : null;
  
  return (
    <div className="camera-container">
      {error ? (
        <div className="error-message">{error}</div>
      ) : (
        <>
          <div className="video-wrapper">
            {/* Vidéo native fluide */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="video-feed"
            />
            
            {/* Image annotée avec masque géométrique en overlay */}
            {annotatedImage && (
              <img 
                src={annotatedImage} 
                alt="Annotated" 
                className="annotated-overlay"
                style={{ opacity: 0.9 }}
              />
            )}
            
            {/* Indicateur de confort/inconfort en haut à gauche */}
            {primaryEmotion && (
              <div className="comfort-indicator">
                {emotionLabel}
              </div>
            )}

            {/* Temperature gauge in top-right */}
            {typeof temperature === 'number' && (
              <div className="temperature-indicator">
                <div className="temp-gauge-mini">
                  <div
                    className="temp-gauge-mini-fill"
                    style={{ height: `${TEMPERATURE_CONFIG.toGaugePercent(temperature)}%` }}
                  />
                </div>
                <span className="temp-value-mini">{TEMPERATURE_CONFIG.format(temperature)}</span>
              </div>
            )}

            {/* Indicateur PPG (confort thermique physiologique) en bas à gauche */}
            {ppgData && ppgState && (
              <div className="ppg-indicator" style={{ borderColor: ppgState.color }}>
                <div className="ppg-header">
                  <span className="ppg-icon">{ppgState.icon}</span>
                  <span className="ppg-label" style={{ color: ppgState.color }}>{ppgState.label}</span>
                </div>
                <div className="ppg-details">
                  <div className="ppg-bar-container">
                    <div
                      className="ppg-bar-fill"
                      style={{
                        width: `${ppgData.pulsatile_intensity * 100}%`,
                        backgroundColor: ppgState.color
                      }}
                    />
                  </div>
                  <span className="ppg-confidence">
                    {ppgData.confidence > 0.5 ? '●' : '○'} {Math.round(ppgData.buffer_fill * 100)}%
                  </span>
                </div>
              </div>
            )}

            {/* Question VLM avec boutons de réponse */}
            {vlmQuestion && (
              <div className="vlm-question-overlay">
                <div className="vlm-question-box">
                  <p className="vlm-question-text">{vlmQuestion}</p>
                  <div className="vlm-action-buttons">
                    {vlmOptions && vlmOptions.length > 0 ? (
                      // Nouveaux boutons avec options dynamiques
                      vlmOptions.map((option, index) => {
                        const optionLower = option.toLowerCase();
                        // Déterminer le type de bouton
                        const isHot = optionLower.includes('chaud') || optionLower === 'baisser';
                        const isCold = optionLower.includes('froid') || optionLower === 'augmenter';
                        const isOk = optionLower.includes('va') || optionLower.includes('bon');

                        return (
                          <button
                            key={index}
                            className={`vlm-button vlm-button-option vlm-button-${
                              isHot ? 'hot' : isCold ? 'cold' : 'ok'
                            }`}
                            onClick={() => onVLMResponse(option)}
                          >
                            {isHot && '🔥 '}
                            {isCold && '❄️ '}
                            {isOk && '✓ '}
                            {option}
                          </button>
                        );
                      })
                    ) : (
                      // Fallback: boutons Oui/Non classiques
                      <>
                        <button
                          className="vlm-button vlm-button-yes"
                          onClick={() => onVLMResponse('oui')}
                        >
                          OUI
                        </button>
                        <button
                          className="vlm-button vlm-button-no"
                          onClick={() => onVLMResponse('non')}
                        >
                          NON
                        </button>
                      </>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {/* Canvas caché pour capture RF-DETR */}
          <canvas ref={canvasRef} style={{ display: 'none' }} />
        </>
      )}
    </div>
  );
};

export default CameraView;
