import numpy as np
import sounddevice as sd
from audio.sample import Sample

class InputStream:
    def __init__(self, rate=44100):
        self.rate = rate

    def capture(self, duration_sec=1):
        """Captures real audio from the default microphone."""
        print("Recording...")
        recording = sd.rec(int(duration_sec * self.rate), 
                           samplerate=self.rate, channels=1)
        sd.wait() # Wait until recording is finished
        return Sample(recording.flatten(), self.rate)

class OutputStream:
    def play(self, sample):
        """Sends the Sample data to your speakers."""
        print(f"Playing {len(sample.data)} samples...")
        sd.play(sample.data, sample.rate)
        sd.wait() # Wait until audio finishes playing