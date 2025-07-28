"""
Unit tests for the language detection module.

This module tests the automatic language detection functionality
for German and English languages.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from language_detection import LanguageDetector, create_language_detector


class TestLanguageDetector:
    """Test cases for the LanguageDetector class."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            'common': {
                'auto_detect_language': True,
                'supported_languages': ['en', 'de'],
                'language_detection_confidence_threshold': 0.7,
                'language': 'en'
            },
            'use_api': False
        }
    
    @pytest.fixture
    def sample_audio_data(self):
        """Generate sample audio data for testing."""
        # Create a simple sine wave as test audio (16kHz, 1 second)
        sample_rate = 16000
        duration = 1.0
        frequency = 440  # A4 note
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        return audio_data
    
    @pytest.fixture
    def mock_whisper_model(self):
        """Mock faster-whisper model for testing."""
        mock_model = Mock()
        # Mock language detection response
        mock_language_info = Mock()
        mock_language_info.language = 'de'
        mock_language_info.language_probability = 0.85
        mock_model.detect_language.return_value = mock_language_info
        return mock_model
    
    @patch('language_detection.ConfigManager')
    def test_init(self, mock_config_manager):
        """Test LanguageDetector initialization."""
        mock_config_manager.get_config_section.return_value = {'common': {}}
        
        detector = LanguageDetector()
        
        assert detector.model is None
        mock_config_manager.get_config_section.assert_called_once_with('model_options')
    
    @patch('language_detection.ConfigManager')
    def test_detect_language_local_success(self, mock_config_manager, mock_whisper_model, sample_audio_data):
        """Test successful local language detection."""
        mock_config_manager.console_print = Mock()
        
        detector = LanguageDetector(mock_whisper_model)
        
        # Test with float32 audio data
        language, confidence = detector.detect_language_local(sample_audio_data)
        
        assert language == 'de'
        assert confidence == 0.85
        mock_whisper_model.detect_language.assert_called_once()
    
    @patch('language_detection.ConfigManager')
    def test_detect_language_local_int16_conversion(self, mock_config_manager, mock_whisper_model):
        """Test local language detection with int16 audio conversion."""
        mock_config_manager.console_print = Mock()
        
        detector = LanguageDetector(mock_whisper_model)
        
        # Create int16 audio data
        int16_audio = np.array([1000, 2000, -1000, -2000], dtype=np.int16)
        
        language, confidence = detector.detect_language_local(int16_audio)
        
        assert language == 'de'
        assert confidence == 0.85
        # Verify the audio was converted to float32 and normalized
        call_args = mock_whisper_model.detect_language.call_args[0][0]
        assert call_args.dtype == np.float32
    
    @patch('language_detection.ConfigManager')
    def test_detect_language_local_error_handling(self, mock_config_manager, mock_whisper_model, sample_audio_data):
        """Test error handling in local language detection."""
        mock_config_manager.console_print = Mock()
        mock_whisper_model.detect_language.side_effect = Exception("Model error")
        
        detector = LanguageDetector(mock_whisper_model)
        
        language, confidence = detector.detect_language_local(sample_audio_data)
        
        assert language == 'en'  # Default fallback
        assert confidence == 0.0
    
    @patch('language_detection.whisper')
    @patch('language_detection.ConfigManager')
    def test_detect_language_api_success(self, mock_config_manager, mock_whisper, sample_audio_data):
        """Test successful API language detection."""
        mock_config_manager.console_print = Mock()
        
        # Mock whisper library
        mock_model = Mock()
        mock_whisper.load_model.return_value = mock_model
        mock_whisper.log_mel_spectrogram.return_value.to.return_value = Mock()
        
        # Mock language detection results
        mock_model.detect_language.return_value = (None, {'de': 0.8, 'en': 0.2})
        
        detector = LanguageDetector()
        
        language, confidence = detector.detect_language_api(sample_audio_data)
        
        assert language == 'de'
        assert confidence == 0.8
    
    @patch('language_detection.ConfigManager')
    def test_detect_language_disabled(self, mock_config_manager, sample_audio_data):
        """Test language detection when disabled in config."""
        config = {
            'common': {
                'auto_detect_language': False,
                'supported_languages': ['en', 'de'],
                'language_detection_confidence_threshold': 0.7,
                'language': 'en'
            }
        }
        mock_config_manager.get_config_section.return_value = config
        
        detector = LanguageDetector()
        
        result = detector.detect_language(sample_audio_data)
        
        assert result is None
    
    @patch('language_detection.ConfigManager')
    def test_detect_language_unsupported_language(self, mock_config_manager, mock_whisper_model, sample_audio_data):
        """Test detection of unsupported language."""
        config = {
            'common': {
                'auto_detect_language': True,
                'supported_languages': ['en', 'de'],
                'language_detection_confidence_threshold': 0.7,
                'language': 'en'
            },
            'use_api': False
        }
        mock_config_manager.get_config_section.return_value = config
        mock_config_manager.console_print = Mock()
        
        # Mock detection of unsupported language (French)
        mock_language_info = Mock()
        mock_language_info.language = 'fr'
        mock_language_info.language_probability = 0.85
        mock_whisper_model.detect_language.return_value = mock_language_info
        
        detector = LanguageDetector(mock_whisper_model)
        
        result = detector.detect_language(sample_audio_data)
        
        assert result is None
    
    @patch('language_detection.ConfigManager')
    def test_detect_language_low_confidence(self, mock_config_manager, mock_whisper_model, sample_audio_data):
        """Test detection with confidence below threshold."""
        config = {
            'common': {
                'auto_detect_language': True,
                'supported_languages': ['en', 'de'],
                'language_detection_confidence_threshold': 0.7,
                'language': 'en'
            },
            'use_api': False
        }
        mock_config_manager.get_config_section.return_value = config
        mock_config_manager.console_print = Mock()
        
        # Mock low confidence detection
        mock_language_info = Mock()
        mock_language_info.language = 'de'
        mock_language_info.language_probability = 0.5  # Below threshold
        mock_whisper_model.detect_language.return_value = mock_language_info
        
        detector = LanguageDetector(mock_whisper_model)
        
        result = detector.detect_language(sample_audio_data)
        
        assert result is None
    
    @patch('language_detection.ConfigManager')
    def test_detect_language_success(self, mock_config_manager, mock_whisper_model, sample_audio_data):
        """Test successful language detection."""
        config = {
            'common': {
                'auto_detect_language': True,
                'supported_languages': ['en', 'de'],
                'language_detection_confidence_threshold': 0.7,
                'language': 'en'
            },
            'use_api': False
        }
        mock_config_manager.get_config_section.return_value = config
        mock_config_manager.console_print = Mock()
        
        # Mock successful detection
        mock_language_info = Mock()
        mock_language_info.language = 'de'
        mock_language_info.language_probability = 0.85
        mock_whisper_model.detect_language.return_value = mock_language_info
        
        detector = LanguageDetector(mock_whisper_model)
        
        result = detector.detect_language(sample_audio_data)
        
        assert result == 'de'
    
    @patch('language_detection.ConfigManager')
    def test_get_fallback_language_configured(self, mock_config_manager):
        """Test fallback language when configured."""
        config = {
            'common': {
                'language': 'de'
            }
        }
        mock_config_manager.get_config_section.return_value = config
        
        detector = LanguageDetector()
        
        result = detector.get_fallback_language()
        
        assert result == 'de'
    
    @patch('language_detection.ConfigManager')
    def test_get_fallback_language_default(self, mock_config_manager):
        """Test fallback language when not configured."""
        config = {
            'common': {
                'language': None
            }
        }
        mock_config_manager.get_config_section.return_value = config
        
        detector = LanguageDetector()
        
        result = detector.get_fallback_language()
        
        assert result == 'en'  # Default fallback
    
    def test_create_language_detector_factory(self):
        """Test the factory function."""
        mock_model = Mock()
        
        detector = create_language_detector(mock_model)
        
        assert isinstance(detector, LanguageDetector)
        assert detector.model == mock_model


class TestLanguageDetectorEdgeCases:
    """Test edge cases and error conditions."""
    
    @patch('language_detection.ConfigManager')
    def test_detect_language_local_no_model(self, mock_config_manager):
        """Test local detection without proper model."""
        mock_config_manager.console_print = Mock()
        
        detector = LanguageDetector("not_a_model")
        audio_data = np.array([1, 2, 3], dtype=np.float32)
        
        with pytest.raises(ValueError, match="Local language detection requires a WhisperModel instance"):
            detector.detect_language_local(audio_data)
    
    @patch('language_detection.ConfigManager')
    def test_detect_language_api_error_handling(self, mock_config_manager):
        """Test API detection error handling."""
        mock_config_manager.console_print = Mock()
        
        detector = LanguageDetector()
        audio_data = np.array([1, 2, 3], dtype=np.float32)
        
        # This should handle the import error gracefully
        with patch('language_detection.whisper') as mock_whisper:
            mock_whisper.load_model.side_effect = Exception("Whisper not available")
            
            language, confidence = detector.detect_language_api(audio_data)
            
            assert language == 'en'  # Default fallback
            assert confidence == 0.0


if __name__ == '__main__':
    pytest.main([__file__])