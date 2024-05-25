import os
import numpy as np
from scipy import signal
from scipy.io import wavfile
from subprocess import CalledProcessError, run


# This function is taken from OpenAI whisper (MIT license)
# See: https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
def load_audio(file: str, sr: int):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def save_audio(y_stereo, path: str, sr: int, force: bool = False):
    """
    Stores a stereo waveform as a .wav file.

    Parameters
    ----------
    y_stereo: (2, N) np.array
        Stereo audio track

    path: str
        Target path for audio

    sr: int
        Sampling rate of y_stereo in Hz

    Returns
    -------
    None
    """

    assert not os.path.exists(path) or force, f"File {path} already exists."

    # Clip to [-1, 1] range, just in case
    y_stereo = np.clip(y_stereo, -1, 1)

    wavfile.write(path, sr, y_stereo.T.astype(np.float32))