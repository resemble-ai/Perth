import unittest
import numpy as np

from perth import DummyWatermarker
from perth.utils import calculate_audio_metrics


class TestDummyWatermarker(unittest.TestCase):
    """Test the DummyWatermarker implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.watermarker = DummyWatermarker()
        # Create a simple sine wave as test audio
        self.sample_rate = 44100
        t = np.linspace(0, 1, self.sample_rate)
        self.test_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    def test_apply_watermark(self):
        """Test that apply_watermark returns an array of the correct shape."""
        watermarked = self.watermarker.apply_watermark(self.test_audio, sample_rate=self.sample_rate)
        self.assertEqual(watermarked.shape, self.test_audio.shape)
    
    def test_get_watermark(self):
        """Test that get_watermark returns a watermark."""
        watermarked = self.watermarker.apply_watermark(self.test_audio, sample_rate=self.sample_rate)
        watermark = self.watermarker.get_watermark(watermarked, sample_rate=self.sample_rate)
        self.assertIsInstance(watermark, np.ndarray)
        self.assertEqual(len(watermark), 32)  # Default length for dummy watermarker
    
    def test_custom_watermark_length(self):
        """Test that get_watermark respects custom watermark length."""
        watermarked = self.watermarker.apply_watermark(self.test_audio, sample_rate=self.sample_rate)
        custom_length = 64
        watermark = self.watermarker.get_watermark(
            watermarked, sample_rate=self.sample_rate, watermark_length=custom_length
        )
        self.assertEqual(len(watermark), custom_length)


class TestAudioMetrics(unittest.TestCase):
    """Test the audio metrics calculation utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple sine wave as test audio
        self.sample_rate = 44100
        t = np.linspace(0, 1, self.sample_rate)
        self.original = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Create a slightly modified version with some noise
        noise = np.random.normal(0, 0.01, len(self.original))
        self.modified = self.original + noise
    
    def test_calculate_metrics(self):
        """Test that audio metrics calculation works correctly."""
        metrics = calculate_audio_metrics(self.original, self.modified)
        
        # Check that metrics are returned and have reasonable values
        self.assertIn('snr', metrics)
        self.assertIn('mse', metrics)
        self.assertIn('psnr', metrics)
        
        # SNR should be positive for this test case
        self.assertGreater(metrics['snr'], 0)
        
        # MSE should be non-zero but small
        self.assertGreater(metrics['mse'], 0)
        self.assertLess(metrics['mse'], 0.1)
        
        # PSNR should be positive and reasonably high
        self.assertGreater(metrics['psnr'], 0)


if __name__ == '__main__':
    unittest.main()