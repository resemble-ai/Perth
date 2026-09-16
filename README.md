<img width="1029" alt="perth-logo-b" src="https://github.com/user-attachments/assets/787a741f-0883-4a92-8732-8c2594b84a3a" />

# Perth

**Perth** is a comprehensive Python library for audio watermarking and detection.

## Overview

Perth enables you to embed imperceptible watermarks in audio files and later detect them, even after the audio has undergone various transformations or manipulations. The library implements multiple watermarking techniques including neural network-based approaches.

## Features

- **Multiple Watermarking Techniques**: Including the Perth-Net Implicit neural network approach
- **Robust Watermarks**: Watermarks can survive common audio transformations like compression, resampling, and more
- **Command-Line Interface**: Easy to use CLI for quick watermarking tasks
- **Python API**: Comprehensive API for integration into your applications
- **Quality Metrics**: Tools to evaluate the perceptual quality of watermarked audio

## Installation

### From PyPI (Recommended)

```bash
pip install resemble-perth
```

Core install is `torch` + `torchaudio` only. `librosa` (and `soxr`) pull LGPL
code and are **not** imported by `perth.perth_net` / `PerthNet`. Install the
optional extra only if you need the CLI or librosa I/O:

```bash
pip install 'resemble-perth[audio]'
```

### From Source

```bash
git clone https://github.com/resemble-ai/Perth
cd Perth
pip install -e .
```

## Quick Start

### Command Line Usage

```bash
# Apply a watermark to an audio file
perth input.wav -o output.wav

# Extract a watermark from an audio file
perth input.wav --extract
```

### Python API Usage

#### Applying a Watermark

```python
import perth
import librosa
import soundfile as sf

# Load audio file
wav, sr = librosa.load("input.wav", sr=None)

# Initialize watermarker
watermarker = perth.PerthImplicitWatermarker()

# Apply watermark
watermarked_audio = watermarker.apply_watermark(wav, watermark=None, sample_rate=sr)

# Save watermarked audio
sf.write("output.wav", watermarked_audio, sr)
```

#### Extracting a Watermark

```python
import perth
import librosa

# Load the watermarked audio
watermarked_audio, sr = librosa.load("output.wav", sr=None)

# Initialize watermarker (same as used for embedding)
watermarker = perth.PerthImplicitWatermarker()

# Extract watermark
watermark = watermarker.get_watermark(watermarked_audio, sample_rate=sr)
print(f"Extracted watermark: {watermark}")
```

### Perth Implicit Watermarker

The Perth-Net Implicit watermarker uses a neural network-based approach for embedding and extracting watermarks. It's designed to be robust against various audio manipulations while maintaining high audio quality.

```python
from perth.perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker

watermarker = PerthImplicitWatermarker(device="cuda")  # Use GPU for faster processing
```

### Dummy Watermarker

A simple placeholder watermarker for testing and demonstration purposes.

```python
from perth import DummyWatermarker

watermarker = DummyWatermarker()
```

## Evaluating Watermarked Audio

The library includes utilities for evaluating the quality and robustness of watermarked audio:

```python
import librosa
from perth.utils import calculate_audio_metrics, plot_audio_comparison

# Load original and watermarked audio
original, sr = librosa.load("input.wav", sr=None)
watermarked, _ = librosa.load("output.wav", sr=None)

# Calculate quality metrics
metrics = calculate_audio_metrics(original, watermarked)
print(f"SNR: {metrics['snr']:.2f} dB")
print(f"PSNR: {metrics['psnr']:.2f} dB")

# Visualize differences
plot_audio_comparison(original, watermarked, sr, output_path="comparison.png")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
