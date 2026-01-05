"""
Utility functions for audio processing and watermarking.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional, Dict
from math import sqrt
from scipy.stats import mode


def _signal_to_frames(data, window_length, pad=True):
    n_samples = data.shape[-1]
    frames = []
    for idx in range(0, n_samples, window_length):
        chunk = data[idx : idx + window_length]
        if pad and chunk.shape[-1] < window_length:
            chunk = np.append(chunk, np.zeros((window_length - chunk.shape[-1])))
        frames.append(chunk)
    return frames


def _frames_to_signal(frames):
    return np.hstack(frames)


def audio_to_raw(wav, bit_depth=16):
    assert wav.dtype.kind == "f", "This function takes floating point arrays"
    unsigned_bit_depth = bit_depth - 1
    range_min, range_max = -(2 ** (unsigned_bit_depth)), 2 ** (unsigned_bit_depth) - 1
    return (wav * range_max).clip(range_min, range_max).astype(np.int16)


def raw_pcm16_tofloat(wav, bit_depth=16):
    unsigned_bit_depth = bit_depth - 1
    range_min, range_max = -(2 ** (unsigned_bit_depth)), 2**unsigned_bit_depth - 1
    a, b = -1.0, 1.0
    return (
        (a + ((wav - range_min) * (b - a)) / (range_max - range_min))
        .clip(-1, 1)
        .astype(np.float32)
    )


def formatted_watermark(watermark_list, length, wrap=True):
    assert len(watermark_list) > 0
    watermark = np.array(watermark_list)
    # Discard extra frames that don't contain the entire watermark.
    # ToDo: Implement Synchronization bits support and return watermark after correlating with synch bits.
    if len(watermark_list) % length:
        watermark = watermark[: -(len(watermark_list) % length)]
    if wrap and len(watermark) > length:
        watermark = np.array(np.split(watermark, len(watermark) // length)).T
        watermark = flatten_watermark(watermark)
    return watermark[:length]


def flatten_watermark(watermark_vector):
    return mode(watermark_vector, axis=1).mode.squeeze(-1)


def modified_binets_fibonnaci(n: int, k: float = 2.5) -> int:
    # This is a modified fibonnaci generator that can generate exponentially spaced sequences like the standard
    # fibonacci series. This function returns the standard fibonnaci sequence using Golden Ratio applying Binet's
    # Formula when k==2 (alpha -> 1.618)
    # The Watermark Accuracy (BER) is slightly more robust for k == 2.5, where alpha -> 1.3

    if n <= 0:
        return 0
    alpha = (1 + sqrt(5)) / k
    beta = (1 - sqrt(5)) / k
    return int(((alpha**n) - (beta**n)) / sqrt(5))


def generate_dummy_watermark(length: int):
    watermark = np.random.random((length,))
    return np.where(watermark > watermark.mean(), 1, 0)


def watermark_str_to_numpy(watermark: str) -> np.ndarray:
    return np.array([int(char) for char in watermark])


def watermark_numpy_to_str(watermark: np.ndarray) -> str:
    return "".join(str(char) for char in watermark)


def validate_string_watermark(watermark: str) -> bool:
    return any([char not in ("1", "0") for char in watermark])


def load_audio(audio_path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Load an audio file using librosa.

    Args:
        audio_path: Path to the audio file
        sr: Target sample rate. If None, the native sample rate is used.

    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        audio, sample_rate = librosa.load(audio_path, sr=sr)
        return audio, sample_rate
    except Exception as e:
        raise IOError(f"Could not load audio file {audio_path}: {e}")


def save_audio(audio_data: np.ndarray, file_path: str, sample_rate: int) -> None:
    """
    Save audio data to a file.

    Args:
        audio_data: Audio data as a numpy array
        file_path: Output file path
        sample_rate: Sample rate for the audio file
    """
    directory = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(directory, exist_ok=True)
    sf.write(file_path, audio_data, sample_rate)


def plot_audio_comparison(
    original: np.ndarray,
    watermarked: np.ndarray,
    sample_rate: int,
    output_path: Optional[str] = None,
) -> None:
    """
    Plot a comparison between original and watermarked audio.

    Args:
        original: Original audio data
        watermarked: Watermarked audio data
        sample_rate: Sample rate of the audio
        output_path: Path to save the plot. If None, plot is shown interactively.
    """
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Plot waveforms
    time = np.arange(len(original)) / sample_rate
    axs[0].plot(time, original, alpha=0.7, label="Original")
    axs[0].plot(time, watermarked, alpha=0.7, label="Watermarked")
    axs[0].set_title("Waveform Comparison")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()

    # Plot difference
    diff = watermarked - original
    axs[1].plot(time, diff)
    axs[1].set_title("Difference (Watermarked - Original)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Difference")

    # Plot spectrogram of difference
    D = librosa.amplitude_to_db(np.abs(librosa.stft(diff)), ref=np.max)
    librosa.display.specshow(D, x_axis="time", y_axis="log", sr=sample_rate, ax=axs[2])
    axs[2].set_title("Spectrogram of Difference")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Frequency (Hz)")
    fig.colorbar(axs[2].collections[0], ax=axs[2], format="%+2.0f dB")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def calculate_audio_metrics(
    original: np.ndarray, watermarked: np.ndarray
) -> Dict[str, float]:
    """
    Calculate audio quality metrics between original and watermarked audio.

    Args:
        original: Original audio data
        watermarked: Watermarked audio data

    Returns:
        Dictionary of quality metrics:
        - snr: Signal-to-Noise Ratio (dB)
        - mse: Mean Squared Error
        - psnr: Peak Signal-to-Noise Ratio (dB)
    """
    if len(original) != len(watermarked):
        raise ValueError("Original and watermarked audio must have the same length")

    # Calculate Mean Squared Error
    mse = np.mean((original - watermarked) ** 2)

    # Calculate Signal-to-Noise Ratio
    signal_power = np.mean(original**2)
    noise_power = mse
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float("inf")

    # Calculate Peak Signal-to-Noise Ratio
    max_value = max(np.max(np.abs(original)), np.max(np.abs(watermarked)))
    psnr = 20 * np.log10(max_value / np.sqrt(mse)) if mse > 0 else float("inf")

    return {"snr": snr, "mse": mse, "psnr": psnr}
