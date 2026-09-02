#!/usr/bin/env python
"""
Advanced example demonstrating watermark robustness testing.

This script applies various audio transformations to watermarked audio 
and tests if the watermark can still be detected after these transformations.
"""
import os
import argparse
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm

from perth import PerthImplicitWatermarker
from perth.utils import calculate_audio_metrics


def apply_mp3_compression(audio, sr, output_path, bitrate='128k'):
    """Apply MP3 compression and decompression to audio."""
    import subprocess
    import tempfile
    
    # Save as WAV
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_wav.close()
    sf.write(temp_wav.name, audio, sr)
    
    # Compress to MP3
    temp_mp3 = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
    temp_mp3.close()
    subprocess.call(['ffmpeg', '-y', '-i', temp_wav.name, '-b:a', bitrate, temp_mp3.name], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Decompress back to WAV
    temp_out = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_out.close()
    subprocess.call(['ffmpeg', '-y', '-i', temp_mp3.name, temp_out.name],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Load the processed audio
    audio_processed, sr = librosa.load(temp_out.name, sr=sr)
    
    # Clean up temporary files
    os.unlink(temp_wav.name)
    os.unlink(temp_mp3.name)
    os.unlink(temp_out.name)
    
    return audio_processed


def apply_transform(audio, sr, transform_type, **kwargs):
    """Apply various transformations to audio."""
    if transform_type == 'mp3':
        bitrate = kwargs.get('bitrate', '128k')
        return apply_mp3_compression(audio, sr, None, bitrate)
    
    elif transform_type == 'resample':
        target_sr = kwargs.get('target_sr', 16000)
        # Resample to target SR
        audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        # Resample back to original SR
        audio_restored = librosa.resample(audio_resampled, orig_sr=target_sr, target_sr=sr)
        return audio_restored
    
    elif transform_type == 'noise':
        noise_level = kwargs.get('noise_level', 0.005)
        noise = np.random.normal(0, noise_level, len(audio))
        return audio + noise
    
    elif transform_type == 'clip':
        clip_level = kwargs.get('clip_level', 0.8)
        return np.clip(audio, -clip_level, clip_level)
    
    elif transform_type == 'reverse':
        # Cut a segment and reverse it
        segment_start = len(audio) // 3
        segment_end = segment_start + len(audio) // 3
        audio_mod = audio.copy()
        audio_mod[segment_start:segment_end] = audio_mod[segment_start:segment_end][::-1]
        return audio_mod
    
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")


def main():
    parser = argparse.ArgumentParser(description="Test watermark robustness against various transformations")
    parser.add_argument("input_file", help="Path to the input audio file to be watermarked")
    parser.add_argument("--output_dir", "-o", default="robustness_results",
                        help="Directory to save results")
    parser.add_argument("--device", "-d", default="cpu", choices=["cpu", "cuda"],
                        help="Device to use for neural network processing")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load audio
    print(f"Loading audio: {args.input_file}")
    audio, sr = librosa.load(args.input_file, sr=None)
    
    # Initialize watermarker
    print("Initializing watermarker...")
    watermarker = PerthImplicitWatermarker(device=args.device)
    
    # Apply watermark
    print("Applying watermark...")
    watermarked_audio = watermarker.apply_watermark(audio, sample_rate=sr)
    
    # Save watermarked audio
    watermarked_path = os.path.join(args.output_dir, "watermarked.wav")
    sf.write(watermarked_path, watermarked_audio, sr)
    print(f"Saved watermarked audio to {watermarked_path}")
    
    # Extract watermark from original watermarked audio (baseline)
    baseline_watermark = watermarker.get_watermark(watermarked_audio, sample_rate=sr)
    baseline_confidence = np.mean(baseline_watermark)
    print(f"Baseline watermark confidence: {baseline_confidence:.4f}")
    
    # Define transformations to test
    transformations = [
        ('mp3', {'bitrate': '128k'}, 'MP3 Compression (128k)'),
        ('mp3', {'bitrate': '64k'}, 'MP3 Compression (64k)'),
        ('resample', {'target_sr': 16000}, 'Resample to 16kHz and back'),
        ('resample', {'target_sr': 8000}, 'Resample to 8kHz and back'),
        ('noise', {'noise_level': 0.001}, 'Low Noise Addition'),
        ('noise', {'noise_level': 0.01}, 'High Noise Addition'),
        ('clip', {'clip_level': 0.8}, 'Amplitude Clipping (0.8)'),
        ('reverse', {}, 'Segment Reversal'),
    ]
    
    # Test each transformation
    results = []
    
    print("\nTesting watermark robustness against transformations:")
    for transform_type, params, label in tqdm(transformations):
        # Apply transformation
        transformed_audio = apply_transform(watermarked_audio, sr, transform_type, **params)
        
        # Save transformed audio
        transformed_path = os.path.join(args.output_dir, f"{transform_type}_transformed.wav")
        sf.write(transformed_path, transformed_audio, sr)
        
        # Extract watermark
        extracted_watermark = watermarker.get_watermark(transformed_audio, sample_rate=sr)
        confidence = np.mean(extracted_watermark)
        
        # Calculate audio quality metrics
        metrics = calculate_audio_metrics(watermarked_audio, transformed_audio)
        
        # Store results
        results.append({
            'transform': label,
            'confidence': confidence,
            'snr': metrics['snr'],
            'success': confidence > 0.75  # Arbitrary threshold for demonstration
        })
        
        print(f"  {label}: Confidence = {confidence:.4f}, SNR = {metrics['snr']:.2f} dB, "
              f"{'PASS' if confidence > 0.75 else 'FAIL'}")
    
    # Plot results
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(results))
    bar_width = 0.35
    
    # Plot confidence scores
    ax.bar(x, [r['confidence'] for r in results], bar_width,
           label='Watermark Confidence', color='skyblue')
    
    # Add threshold line
    ax.axhline(y=0.75, linestyle='--', color='red', alpha=0.7, 
               label='Success Threshold (0.75)')
    
    # Add labels and title
    ax.set_xlabel('Transformation')
    ax.set_ylabel('Watermark Confidence')
    ax.set_title('Watermark Robustness to Various Transformations')
    ax.set_xticks(x)
    ax.set_xticklabels([r['transform'] for r in results], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'robustness_results.png'))
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()