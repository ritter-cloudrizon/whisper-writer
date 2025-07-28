"""
Language detection module for automatic language identification in audio.

This module provides functionality to automatically detect the spoken language
in audio data using Whisper's built-in language detection capabilities.
"""

import numpy as np
from typing import Tuple, Optional
from faster_whisper import WhisperModel

from utils import ConfigManager

# Import whisper only when needed for API detection
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class LanguageDetector:
    """Handles automatic language detection for audio transcription."""
    
    def __init__(self, model=None):
        """
        Initialize the language detector.
        
        Args:
            model: Pre-initialized Whisper model (local or API).
        """
        self.model = model
        self._config = ConfigManager.get_config_section('model_options')
    
    def detect_language_local(self, audio_data: np.ndarray) -> Tuple[str, float]:
        """
        Detect language using local Whisper model.
        
        Args:
            audio_data: Audio data as numpy array (float32, 16kHz).
            
        Returns:
            Tuple of (language_code, confidence_score).
        """
        if not isinstance(self.model, WhisperModel):
            raise ValueError("Local language detection requires a WhisperModel instance")
        
        # Convert int16 to float32 if needed
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        
        try:
            # Use faster-whisper's detect_language method
            language_info = self.model.detect_language(audio_data)
            
            # Get the most probable language
            if hasattr(language_info, 'language'):
                detected_lang = language_info.language
                confidence = getattr(language_info, 'language_probability', 0.0)
            else:
                # Fallback for different API versions
                detected_lang = language_info[0] if isinstance(language_info, tuple) else 'en'
                confidence = language_info[1] if isinstance(language_info, tuple) and len(language_info) > 1 else 0.0
            
            ConfigManager.console_print(f'Detected language: {detected_lang} (confidence: {confidence:.2f})')
            return detected_lang, confidence
            
        except Exception as e:
            ConfigManager.console_print(f'Language detection failed: {e}')
            return 'en', 0.0  # Default fallback
    
    def detect_language_api(self, audio_data: np.ndarray) -> Tuple[str, float]:
        """
        Detect language using OpenAI API.
        
        Note: OpenAI API doesn't provide separate language detection,
        so we'll use a workaround with standard whisper library.
        
        Args:
            audio_data: Audio data as numpy array.
            
        Returns:
            Tuple of (language_code, confidence_score).
        """
        if not WHISPER_AVAILABLE:
            ConfigManager.console_print('Standard whisper library not available for API language detection')
            return 'en', 0.0
            
        try:
            # Use standard whisper for language detection
            model = whisper.load_model("base")  # Use small model for detection
            
            # Convert int16 to float32 if needed
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Pad or trim audio to 30 seconds (whisper's expected length)
            expected_length = 16000 * 30  # 30 seconds at 16kHz
            if len(audio_data) > expected_length:
                audio_data = audio_data[:expected_length]
            elif len(audio_data) < expected_length:
                audio_data = np.pad(audio_data, (0, expected_length - len(audio_data)))
            
            # Create mel spectrogram
            mel = whisper.log_mel_spectrogram(audio_data).to(model.device)
            
            # Detect language
            _, probs = model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)
            confidence = probs[detected_lang]
            
            ConfigManager.console_print(f'API language detection: {detected_lang} (confidence: {confidence:.2f})')
            return detected_lang, confidence
            
        except Exception as e:
            ConfigManager.console_print(f'API language detection failed: {e}')
            return 'en', 0.0  # Default fallback
    
    def detect_language(self, audio_data: np.ndarray) -> Optional[str]:
        """
        Detect the language of the given audio data.
        
        Args:
            audio_data: Audio data as numpy array.
            
        Returns:
            Detected language code or None if detection disabled/failed.
        """
        # Check if auto-detection is enabled
        if not self._config['common']['auto_detect_language']:
            return None
        
        supported_languages = self._config['common']['supported_languages']
        confidence_threshold = self._config['common']['language_detection_confidence_threshold']
        
        try:
            # Choose detection method based on model type
            if self._config['use_api']:
                detected_lang, confidence = self.detect_language_api(audio_data)
            else:
                detected_lang, confidence = self.detect_language_local(audio_data)
            
            # Check if detected language is supported
            if detected_lang not in supported_languages:
                ConfigManager.console_print(f'Detected language {detected_lang} not in supported list {supported_languages}')
                return None
            
            # Check confidence threshold
            if confidence < confidence_threshold:
                ConfigManager.console_print(f'Language detection confidence {confidence:.2f} below threshold {confidence_threshold}')
                return None
            
            return detected_lang
            
        except Exception as e:
            ConfigManager.console_print(f'Language detection error: {e}')
            return None
    
    def get_fallback_language(self) -> str:
        """
        Get the fallback language when detection fails.
        
        Returns:
            Language code to use as fallback.
        """
        configured_lang = self._config['common']['language']
        if configured_lang:
            return configured_lang
        
        # Default to English if no language configured
        return 'en'


def create_language_detector(model=None) -> LanguageDetector:
    """
    Factory function to create a language detector instance.
    
    Args:
        model: Pre-initialized Whisper model.
        
    Returns:
        LanguageDetector instance.
    """
    return LanguageDetector(model)