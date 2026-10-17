# record-clip

import sounddevice as sd 
from scipy.io.wavfile import write


fs = 44100 # Sample rate 
seconds = 5 # Duration of recording

print("Recording...")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait() # wair until recording is finishe
write("output.wav", fs, myrecording)
print("Saved as output.wav")