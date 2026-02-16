# audio/effects.py
import numpy as np

class AudioTransformer:
    def process(self, data, rate):
        """Processes raw numpy data and returns raw numpy data."""
        # Chain your effects
        data = self.distortion(data, gain=2.0)
        data = self.reverb(data, delay_ms=100, decay=0.4, sample_rate=rate)
        
        return data  # Returning raw numpy array

    def stretch(self, data, factor, retain_pitch=False):
        # ... (Your stretch logic)
        return data

    def reverb(self, data, delay_ms, decay, sample_rate):
        delay_samples = int((delay_ms / 1000) * sample_rate)
        out = np.copy(data)
        # Simple feedback loop
        for i in range(delay_samples, len(data)):
            out[i] += out[i - delay_samples] * decay
        return out

    def distortion(self, data, gain):
        return np.tanh(data * gain)