import React from 'react';
import { TEMPERATURE_CONFIG } from '../config/temperature';
import './CameraView.css';

const CameraView = ({ videoRef, canvasRef, annotatedImage, emotion, error, vlmQuestion, vlmOptions, onVLMResponse, temperature, primaryEmotion }) => {
  // Afficher l'émotion du détecteur primary (FER): 'confortable' ou 'inconfortable'
  // Capitaliser la première lettre pour l'affichage
  const emotionLabel = primaryEmotion 
    ? primaryEmotion.charAt(0).toUpperCase() + primaryEmotion.slice(1)
    : '';
  
  // Debug
  console.log('CameraView primaryEmotion:', { primaryEmotion, emotionLabel });
  
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
