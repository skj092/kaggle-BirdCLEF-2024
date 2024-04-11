# Convert sound of bird to spectrogram
from fastai.vision.all import Path, get_files
import soundfile as sf
import librosa as lb
import librosa.display as lbd
from soundfile import SoundFile
import numpy as np


class Config:
    sampling_rate = 32000
    duration = 5
    fmin = 0
    fmax = None
    audios_path = Path("/kaggle/input/birdclef-2023/train_audio")
    out_dir_train = Path("specs/train")

    out_dir_valid = Path("specs/valid")


def get_audio_info(filepath):
    """Get some properties from  an audio file"""
    with SoundFile(filepath) as f:
        sr = f.samplerate
        frames = f.frames
        duration = float(frames) / sr
    return {"frames": frames, "sr": sr, "duration": duration}


def compute_melspec(y, sr, n_mels, fmin, fmax):
    """
    Computes a mel-spectrogram and puts it at decibel scale
    Arguments:
        y {np array} -- signal
        params {AudioParams} -- Parameters to use for the spectrogram. Expected to have the attributes sr, n_mels, f_min, f_max
    Returns:
        np array -- Mel-spectrogram
    """
    melspec = lb.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )

    melspec = lb.power_to_db(melspec).astype(np.float32)
    return melspec


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V


def main():
    path = Path("data/")
    audio_files = get_files(path / "train_audio", extensions=".ogg")
    print(f"Found {len(audio_files)} audio files")
    audio = audio_files[0]
    info = get_audio_info(audio)
    print(info)
    y, sr = sf.read(audio)
    # play audio
    lbd.Audio(y, sr=sr)


if __name__ == "__main__":
    main()
