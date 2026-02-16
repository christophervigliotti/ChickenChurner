import numpy as np

class AudioTransformer:
    def process(self, data, rate):
        """Standard DSP pipeline."""
        # data = self.stretch(data, factor=1.5) # Example: 1.5x slower, lower pitch
        data = self.distortion(data, gain=1.5)
        data = self.reverb(data, delay_ms=150, decay=0.3, sample_rate=rate)
        return data

    def stretch(self, data, factor):
        """
        Resamples audio. Pitch and Speed are linked.
        factor > 1.0: Slower and Lower Pitch
        factor < 1.0: Faster and Higher Pitch
        """
        if factor == 1.0:
            return data
            
        # Create a new time-axis based on the stretch factor
        new_indices = np.arange(0, len(data), factor)
        # Map the original data onto the new indices
        return np.interp(new_indices, np.arange(len(data)), data)

    def reverb(self, data, delay_ms, decay, sample_rate):
        delay_samples = int((delay_ms / 1000) * sample_rate)
        out = np.copy(data)
        for i in range(delay_samples, len(data)):
            out[i] += out[i - delay_samples] * decay
        return out

    def distortion(self, data, gain):
        return np.tanh(data * gain)