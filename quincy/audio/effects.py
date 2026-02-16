import numpy as np
from audio.sample import Sample

class AudioTransformer:
    def process(self, sample):
        """Pipeline entry point."""
        data = sample.data
        
        # Example: Apply effects in sequence
        data = self.distortion(data, gain=2.5)
        data = self.reverb(data, delay_ms=100, decay=0.3, sample_rate=sample.rate)
        
        # Example: Slow down by 50% without pitch correction
        # data = self.stretch(data, factor=1.5, retain_pitch=False)
        
        return Sample(data, sample.rate)

    def stretch(self, data, factor, retain_pitch=False):
        """
        factor > 1.0 = Slower
        factor < 1.0 = Faster
        """
        if retain_pitch:
            # Note: True pitch-retained stretching requires 
            # an FFT-based Phase Vocoder (complex math).
            print("Pitch-retained stretching not implemented yet.")
            return data
        else:
            # Linear Resampling (Changes pitch like a vinyl record)
            new_length = int(len(data) * factor)
            return np.interp(
                np.linspace(0, len(data), new_length), 
                np.arange(len(data)), 
                data
            )

    def reverb(self, data, delay_ms, decay, sample_rate):
        """Simple Feedback Delay (Comb Filter)"""
        delay_samples = int((delay_ms / 1000) * sample_rate)
        out = np.copy(data)
        for i in range(delay_samples, len(data)):
            out[i] += out[i - delay_samples] * decay
        return out

    def distortion(self, data, gain):
        """Soft-clipping using Hyperbolic Tangent"""
        return np.tanh(data * gain)