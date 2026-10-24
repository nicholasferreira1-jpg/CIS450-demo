# W9B.py: Normalize, Trim Silence, 2X Speed Up
# LICENSE = numpy (BSD-3-Clause), scipy (BSD-3-Clause)
# Both permissive -> compatible with a closed-source commercial application.
# Replaces pedalboard (GPLv3), which is NOT compatible with closed-source use.
import numpy as np
from scipy.io import wavfile


# trim starting and ending silence from the audio clip
def trim_silence(audio):
    chunk_size = 1024  # Process in 1024-sample chunks (23ms at 44.1kHz)
    threshold = 0.1    # linear fraction of silence compared to max amplitude
    start = 0
    for i in range(0, len(audio), chunk_size):
        if np.max(np.abs(audio[i:i + chunk_size])) > threshold:
            start = i
            break
    end = len(audio)
    for i in range(len(audio) - chunk_size, 0, -chunk_size):
        if np.max(np.abs(audio[i:i + chunk_size])) > threshold:
            end = i + chunk_size
            break
    return audio[start:end]


# Normalize audio to -1.0 to 1.0 range
def normalize_audio(audio):
    peak = np.max(np.abs(audio))
    if peak > 0:
        return audio / peak
    return audio


# Change audio speed
def change_speed(audio, rate):
    indices = (np.arange(0, len(audio) - 1, rate)).astype(int)
    return audio[indices]


# Convert integer PCM samples to float in the -1.0 to 1.0 range
def to_float(audio):
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        if info.min == 0:  # 8-bit WAV is unsigned
            return (audio.astype(np.float32) - 128) / 128.0
        return audio.astype(np.float32) / max(abs(info.min), info.max)
    return audio.astype(np.float32)


# Read the input file (scipy replaces pedalboard's AudioFile)
samplerate, audio = wavfile.read("output.wav")
if audio.ndim > 1:
    audio = audio[:, 0]  # first channel
audio = to_float(audio)

normalized = normalize_audio(audio)
trimmed = trim_silence(normalized)
sped_up = change_speed(trimmed, 2.0)

# Write output as 16-bit PCM (scipy replaces pedalboard's AudioFile)
out = np.clip(sped_up, -1.0, 1.0)
wavfile.write("processed.wav", samplerate, (out * 32767).astype(np.int16))
print("Saved processed.wav")