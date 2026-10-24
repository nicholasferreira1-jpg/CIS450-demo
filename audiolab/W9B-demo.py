# W9B-demo.py: Normalize, Trim Silence, 2X Speed Up: LICENSE = numpy (BSD-3-Clause), pedalboard (GPLv3)
import numpy as np
from pedalboard.io import AudioFile

# trim starting and ending silence from the audio clip
def trim_silence(audio):
    chunk_size = 1024  # Process in 1024-sample chunks (23ms at 44.1kHz)
    threshold = 0.1    # linear fraction of silence compared to max amplitude
    # Find start index
    start = 0
    for i in range(0, len(audio), chunk_size):
        if np.max(np.abs(audio[i:i+chunk_size])) > threshold:
            start = i
            break
    # Find end index
    end = len(audio)
    for i in range(len(audio) - chunk_size, 0, -chunk_size):
        if np.max(np.abs(audio[i:i+chunk_size])) > threshold:
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

# Read the input file
with AudioFile("output.wav") as f:
    samplerate = f.samplerate
    audio = f.read(f.frames)
if audio.ndim > 1:
    audio = audio[0]  # select the first (and only) channel

# Normalize
normalized = normalize_audio(audio)

# Trim silence
trimmed = trim_silence(normalized)

# Speed up 2x
sped_up = change_speed(trimmed, 2.0)

# Write output
with AudioFile("processed.wav", 'w', samplerate, 1) as out:
    out.write(sped_up)
print("Saved processed.wav")