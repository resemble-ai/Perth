# Borrowed from Resembletron

import logging
import tempfile
import warnings
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
from audioread import NoBackendError
from pydub import AudioSegment


logger = logging.getLogger(__name__)


class WatermarkingException(Exception):
    pass


class CorruptedAudioException(Exception):
    pass


def load_wav(fpath, target_sr, res_algo="kaiser_best"):
    """
    :param target_sr: expected sample rate after loading and possibly resampling. If None,
    there will be no resampling.
    :param res_algo: algorithm for resampling. If None, there will also be no resampling but if
    the target_sr is valid, the actual sample rate of the audio on disk will be checked against
    and an error will be thrown if they do not match.
    """
    bit_depth = sf.SoundFile(str(fpath)).subtype
    if not bit_depth.startswith("PCM"):
        raise WatermarkingException(
            "Unsupported Audio type for Watermarking. "
            "Only 16 or 24-bit PCM/WAV/AIFF audio files can be watermarked."
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wav, actual_sr = librosa.core.load(
                str(fpath), sr=(target_sr if res_algo else None), res_type=res_algo
            )
    except (EOFError, NoBackendError):
        raise CorruptedAudioException("Failed to load audio file")

    if target_sr is not None:
        assert actual_sr == target_sr, (
            "Loaded audio doesn't have expected sampling rate (%s vs "
            "%s, resampling_algo=%s)" % (actual_sr, target_sr, res_algo)
        )

    return wav, actual_sr


def save_wav(wav, file_or_path, sample_rate: int, subtype="PCM_16"):
    """
    :param wav: a float32 numpy array
    """
    assert wav.dtype.kind == "f", "This function takes floating point arrays"

    # Float32 -> PCM_16 conversion
    if subtype == "PCM_16":
        range_min, range_max = -(2**15), 2**15 - 1
        wav = (wav * range_max).clip(range_min, range_max).astype(np.int16)

    file_or_path = str(file_or_path) if isinstance(file_or_path, Path) else file_or_path
    sf.write(file_or_path, wav, sample_rate, subtype=subtype, format="wav")


def pitch_shift(wav, sample_rate, semitones):  # TODO: unused function?
    import pyrubberband as pyrb

    return pyrb.pitch_shift(wav, sample_rate, semitones)


def convert_to_mp3(wav_path, sample_rate=22050):
    segment = AudioSegment.from_wav(wav_path)
    tmpfile = tempfile.SpooledTemporaryFile(suffix=".mp3")
    segment = segment.set_frame_rate(sample_rate)
    segment.export(tmpfile, bitrate="48k", format="mp3")
    return tmpfile
