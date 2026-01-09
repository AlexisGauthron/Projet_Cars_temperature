import React, { useState, useEffect, useCallback, useRef } from 'react';
import CameraView from './components/CameraView';
import ModeToggle from './components/ModeToggle';
import { useCamera } from './hooks/useCamera';
import { api } from './services/api';
import { TIMING_CONFIG } from './config/timing';
import { TEMPERATURE_CONFIG } from './config/temperature';
import './App.css';

function App() {
  const { videoRef, canvasRef, captureFrame, error } = useCamera();

  const [temperature, setTemperature] = useState(TEMPERATURE_CONFIG.DEFAULT);
  const temperatureRef = useRef(temperature);
  const [currentEmotion, setCurrentEmotion] = useState('');
  const [annotatedImage, setAnnotatedImage] = useState(null);
  const [primaryEmotion, setPrimaryEmotion] = useState('');
  const [vlmQuestion, setVlmQuestion] = useState(null);
  const [vlmOptions, setVlmOptions] = useState(null);
  const [isWaitingVLM, setIsWaitingVLM] = useState(false);
  const [lastVLMCheck, setLastVLMCheck] = useState(Date.now());
  const [detectionMode, setDetectionMode] = useState('single'); // 'single' ou 'multi'
  const [facesSummary, setFacesSummary] = useState(null);

  // Synchroniser la ref avec le state
  useEffect(() => {
    temperatureRef.current = temperature;
  }, [temperature]);

  // 🤖 ANALYSE RF-DETR (10 FPS)
  const processFrame = useCallback(async () => {
    const frameData = captureFrame();
    if (!frameData) return;

    try {
      const result = await api.sendFrame(frameData, temperatureRef.current, detectionMode);

      setCurrentEmotion(result.emotion);
      setAnnotatedImage(result.annotated_image);

      // Mettre à jour l'émotion du détecteur primary (FER) si disponible
      if (result.primary_emotion !== undefined) {
        setPrimaryEmotion(result.primary_emotion);
      }

      // Mettre à jour le résumé des visages
      if (result.faces_summary) {
        setFacesSummary(result.faces_summary);
      }

      // Mettre à jour la température avec celle du backend !
      if (result.temperature !== undefined) {
        setTemperature(result.temperature);
      }
    } catch (err) {
      console.error('Erreur traitement frame:', err);
    }
  }, [captureFrame, detectionMode]);

  // 🧠 VLM Check
  const checkVLM = useCallback(async () => {
    if (isWaitingVLM) return;

    const elapsed = Date.now() - lastVLMCheck;
    if (elapsed < TIMING_CONFIG.VLM_COOLDOWN_MS) return;

    try {
      const result = await api.checkVLM();

      if (result.question) {
        setVlmQuestion(result.question);
        setVlmOptions(result.options || null);
        setIsWaitingVLM(true);
      } else {
        setLastVLMCheck(Date.now());
      }
    } catch (err) {
      console.error('[VLM Check] Erreur:', err);
    }
  }, [isWaitingVLM, lastVLMCheck]);

  const handleVLMResponse = async (response) => {
    console.log('[VLM Response] User clicked:', response);

    try {
      const result = await api.sendVLMResponse(response);

      setVlmQuestion(null);
      setVlmOptions(null);
      setIsWaitingVLM(false);
      setLastVLMCheck(Date.now());

      // Mettre à jour la température si elle a changé
      if (result.new_temperature !== null && result.new_temperature !== undefined) {
        setTemperature(result.new_temperature);
      }

      console.log('[VLM Response] Response sent to backend');
    } catch (err) {
      console.error('[VLM Response] Error:', err);
    }
  };

  const handleModeChange = (newMode) => {
    setDetectionMode(newMode);
    setFacesSummary(null); // Reset summary when changing mode
  };

  // Analyse des frames (5 FPS par défaut - configurable dans config/timing.js)
  useEffect(() => {
    const frameInterval = setInterval(processFrame, TIMING_CONFIG.FRAME_INTERVAL_MS);
    return () => clearInterval(frameInterval);
  }, [processFrame]);

  // VLM Check (toutes les 2s par défaut - configurable dans config/timing.js)
  useEffect(() => {
    const vlmInterval = setInterval(checkVLM, TIMING_CONFIG.VLM_CHECK_INTERVAL_MS);
    return () => clearInterval(vlmInterval);
  }, [checkVLM]);

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <div className="logo-space">
            <img src="/Stellantis.png" alt="Stellantis" />
          </div>
          <h1 className="app-title">CARE</h1>
          <ModeToggle mode={detectionMode} onModeChange={handleModeChange} />
        </div>
      </header>

      <div className="app-content">
        <CameraView
          videoRef={videoRef}
          canvasRef={canvasRef}
          annotatedImage={annotatedImage}
          emotion={currentEmotion}
          error={error}
          vlmQuestion={vlmQuestion}
          vlmOptions={vlmOptions}
          onVLMResponse={handleVLMResponse}
          temperature={temperature}
          primaryEmotion={primaryEmotion}
          facesSummary={facesSummary}
          detectionMode={detectionMode}
        />
      </div>
    </div>
  );
}

export default App;
