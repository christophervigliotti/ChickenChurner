# listening: constant
# transformation: applied to incoming audio signal
# output: constant

import sounddevice as sd
import numpy as np
from scipy.interpolate import interp1d
import queue
import time

class SmartAudioProcessor:
    def __init__(self, sample_rate=44100, segment_duration=3):
        self.fs = sample_rate
        self.segment_len = int(sample_rate * segment_duration)
        self.buffer = queue.Queue()
        self.out_blocksize = 1024
        
        # Threshold for "50% utilization" (half of the 3s segment duration)
        self.limit_threshold = segment_duration * 0.5 

    def stretch_audio(self, audio_data, factor=1.19):
        n_samples = len(audio_data)
        new_n_samples = int(n_samples * factor)
        old_indices = np.linspace(0, n_samples - 1, n_samples)
        new_indices = np.linspace(0, n_samples - 1, new_n_samples)
        
        interpolator = interp1d(old_indices, audio_data, axis=0, kind='linear')
        return interpolator(new_indices).astype(np.float32)

    def add_reverb(self, audio_data):
        delay = int(self.fs * 0.15)
        decay = 0.4
        out = np.zeros_like(audio_data)
        out[delay:] = audio_data[:-delay] * decay
        return (audio_data + out) * 0.6

    def input_callback(self, indata, frames, time_info, status):
        start_time = time.time()
        
        # --- THE TRANSFORMATION ---
        stretched = self.stretch_audio(indata.copy(), factor=1.19)
        #transformed = self.add_reverb(stretched)
        transformed = stretched
        
        # --- UTILIZATION CHECK ---
        processing_duration = time.time() - start_time
        utilization = (processing_duration / 3.0) * 100
        
        if processing_duration > self.limit_threshold:
            print(f"⚠️ LIMITER ENGAGED: Utilized {utilization:.1f}%. Flushing buffer to sync.")
            # Clear the queue to prevent massive drift
            while not self.buffer.empty():
                try: self.buffer.get_nowait()
                except queue.Empty: break
        
        # Feed the output queue
        for sample in transformed:
            self.buffer.put(sample)

    def output_callback(self, outdata, frames, time_info, status):
        for i in range(frames):
            try:
                outdata[i] = self.buffer.get_nowait()
            except queue.Empty:
                outdata[i] = 0 # Silence if we run out of audio

    def run(self):
        print(f"Monitoring load. Limit: {self.limit_threshold}s processing time.")
        
        with sd.InputStream(channels=1, samplerate=self.fs, 
                            blocksize=self.segment_len, 
                            callback=self.input_callback):
            
            with sd.OutputStream(channels=1, samplerate=self.fs, 
                                 blocksize=self.out_blocksize, 
                                 callback=self.output_callback):
                
                print("Live. Press Ctrl+C to stop.")
                while True:
                    sd.sleep(1000)

if __name__ == "__main__":
    proc = SmartAudioProcessor()
    try:
        proc.run()
    except KeyboardInterrupt:
        print("\nExit.")