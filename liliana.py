# alternates between sampling the mic and sampling the output buffer
# gaps: yes
import sounddevice as sd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter
import queue
import time

class LoFiFeedbackProcessor:
    def __init__(self, sample_rate=44100, segment_duration=3):
        self.fs = sample_rate
        self.segment_len = int(sample_rate * segment_duration)
        self.buffer = queue.Queue()
        
        # Tape Memory
        self.last_output_segment = np.zeros((self.segment_len, 1), dtype=np.float32)
        self.sample_from_mic = True
        self.limit_threshold = segment_duration * 0.5 

    def low_pass_filter(self, data, cutoff=2500):
        """Removes harsh high frequencies from the feedback loop."""
        nyquist = 0.5 * self.fs
        normal_cutoff = cutoff / nyquist
        # 2nd order Butterworth for a smooth roll-off
        b, a = butter(2, normal_cutoff, btype='low', analog=False)
        return lfilter(b, a, data, axis=0)

    def stretch_audio(self, audio_data, factor=1.19):
        data_flat = audio_data.flatten()
        n_samples = len(data_flat)
        new_n_samples = int(n_samples * factor)
        
        old_indices = np.linspace(0, n_samples - 1, n_samples)
        new_indices = np.linspace(0, n_samples - 1, new_n_samples)
        
        # Fast interpolation
        stretched = np.interp(new_indices, old_indices, data_flat)
        return stretched.astype(np.float32).reshape(-1, 1)

    def add_reverb(self, audio_data):
        delay = int(self.fs * 0.15)
        decay = 0.4
        out = np.zeros_like(audio_data)
        if len(audio_data) > delay:
            out[delay:] = audio_data[:-delay] * decay
        return (audio_data + out) * 0.6

    def input_callback(self, indata, frames, time_info, status):
        start_time = time.time()
        
        if self.sample_from_mic:
            source_material = indata.copy()
            source_name = "MIC"
        else:
            # Apply Low-Pass Filter ONLY to the feedback loop
            source_material = self.low_pass_filter(self.last_output_segment.copy())
            source_name = "FILTERED FEEDBACK"
        
        print(f"Source: {source_name} | Toggle: {self.sample_from_mic}")
        
        # Process: Stretch -> Reverb
        stretched = self.stretch_audio(source_material, factor=1.19)
        transformed = self.add_reverb(stretched)
        
        # Utilization Check
        proc_time = time.time() - start_time
        if proc_time > self.limit_threshold:
            print(f"⚠️ Limiter: { (proc_time/3)*100 :.1f}% load. Flushing.")
            while not self.buffer.empty():
                try: self.buffer.get_nowait()
                except queue.Empty: break
        
        for sample in transformed:
            self.buffer.put(sample)
            
        self.sample_from_mic = not self.sample_from_mic

    def output_callback(self, outdata, frames, time_info, status):
        for i in range(frames):
            try:
                val = self.buffer.get_nowait()
                outdata[i] = val
                # Keep the "tape" rolling
                self.last_output_segment = np.roll(self.last_output_segment, -1, axis=0)
                self.last_output_segment[-1] = val
            except queue.Empty:
                outdata[i] = 0

    def run(self):
        with sd.InputStream(channels=1, samplerate=self.fs, 
                            blocksize=self.segment_len, 
                            callback=self.input_callback):
            with sd.OutputStream(channels=1, samplerate=self.fs, 
                                 blocksize=1024, 
                                 callback=self.output_callback):
                print("Running... Audio will alternate and darken over time.")
                while True:
                    sd.sleep(1000)

if __name__ == "__main__":
    proc = LoFiFeedbackProcessor()
    try:
        proc.run()
    except KeyboardInterrupt:
        print("\nExit.")