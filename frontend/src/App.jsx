import React, { useState, useEffect, useCallback, useRef } from 'react';
import CameraView from './components/CameraView';
import { useCamera } from './hooks/useCamera';
import { api } from './services/api';
import './App.css';

function App() {
  const { videoRef, canvasRef, captureFrame, error } = useCamera();
  
  const [temperature, setTemperature] = useState(20);
  const temperatureRef = useRef(temperature);
  const [currentEmotion, setCurrentEmotion] = useState('');
  const [annotatedImage, setAnnotatedImage] = useState(null);
  const [primaryEmotion, setPrimaryEmotion] = useState('');
  const [vlmQuestion, setVlmQuestion] = useState(null);
  const [isWaitingVLM, setIsWaitingVLM] = useState(false);
  const [lastVLMCheck, setLastVLMCheck] = useState(Date.now());

  // Synchroniser la ref avec le state
  useEffect(() => {
    temperatureRef.current = temperature;
  }, [temperature]);

  // 🤖 ANALYSE RF-DETR (10 FPS)
  const processFrame = useCallback(async () => {
    const frameData = captureFrame();
    if (!frameData) return;

    try {
      const result = await api.sendFrame(frameData, temperatureRef.current);
      
      setCurrentEmotion(result.emotion);
      setAnnotatedImage(result.annotated_image);
      
      // Debug: voir ce qui arrive du backend
      console.log('Backend result:', result);
      console.log('Primary emotion received:', result.primary_emotion);
      
      // Mettre à jour l'émotion du détecteur primary (FER) si disponible
      if (result.primary_emotion !== undefined) {
        setPrimaryEmotion(result.primary_emotion);
        console.log('Primary emotion (FER) updated to:', result.primary_emotion);
      } else {
        console.warn('No primary_emotion in backend result');
      }
      
      // Mettre à jour la température avec celle du backend !
      if (result.temperature !== undefined) {
        setTemperature(result.temperature);
      }
    } catch (err) {
      console.error('Erreur traitement frame:', err);
    }
  }, [captureFrame]);

  // 🧠 VLM Check
  const checkVLM = useCallback(async () => {
    if (isWaitingVLM) return;

    const elapsed = Date.now() - lastVLMCheck;
    if (elapsed < 5000) return;

    try {
      const result = await api.checkVLM();
      
      if (result.question) {
        setVlmQuestion(result.question);
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
      // Envoyer la réponse au backend (le backend gérera l'ajustement progressif)
      await api.sendVLMResponse(response);
      
      setVlmQuestion(null);
      setIsWaitingVLM(false);
      setLastVLMCheck(Date.now());
      
      console.log('[VLM Response] Response sent to backend, temperature will be adjusted server-side');
    } catch (err) {
      console.error('[VLM Response] Error:', err);
    }
  };

  // RF-DETR Analysis à 20 FPS
  useEffect(() => {
    const frameInterval = setInterval(processFrame, 50);
    return () => clearInterval(frameInterval);
  }, [processFrame]);

  // VLM Check toutes les secondes
  useEffect(() => {
    const vlmInterval = setInterval(checkVLM, 1000);
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
          <div></div>
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
          onVLMResponse={handleVLMResponse}
          temperature={temperature}
          primaryEmotion={primaryEmotion}
        />
        
        {/* Slider de température enlevé - la température change automatiquement */}
      </div>
    </div>
  );
}

export default App;
