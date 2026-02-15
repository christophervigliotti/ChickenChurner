# ts controls the number and length of the capture layers
import sounddevice as sd
import numpy as np
import threading
import queue
import time
import random
from scipy.signal import butter, lfilter

class LayerThread(threading.Thread):
    def __init__(self, layer_id, source_type, duration_range, fs, mixer_queue, processor):
        super().__init__(daemon=True)
        self.layer_id = layer_id
        self.source_type = source_type
        # duration_range is now a tuple: (min, max)
        self.min_dur, self.max_dur = duration_range
        self.fs = fs
        self.mixer_queue = mixer_queue
        self.processor = processor

    def apply_fade(self, audio, fade_len=2000):
        if len(audio) < fade_len: return audio
        fade_out = np.linspace(1., 0., fade_len)
        audio[-fade_len:] *= fade_out
        return audio

    def stretch_and_verb(self, data):
        if data.size == 0 or np.max(np.abs(data)) < 0.005: 
            return None
        
        n_samples = len(data)
        new_indices = np.linspace(0, n_samples - 1, int(n_samples * 1.19))
        stretched = np.interp(new_indices, np.arange(n_samples), data.flatten())
        
        delay = int(self.fs * 0.15)
        out = np.zeros_like(stretched)
        if len(stretched) > delay:
            out[delay:] = stretched[:-delay] * 0.4
        
        combined = (stretched + out) * 0.4
        return self.apply_fade(combined.astype(np.float32))

    def run(self):
        print(f"Layer {self.layer_id} [{self.source_type}] active. Randomizing between {self.min_dur}-{self.max_dur}s")
        while True:
            # Pick a new random interval for this specific loop
            current_interval = random.uniform(self.min_dur, self.max_dur)
            time.sleep(current_interval)
            
            data = self.processor.get_source_data(self.source_type)
            processed = self.stretch_and_verb(data)
            if processed is not None:
                self.mixer_queue.put(processed)

class MultiLayerProcessor:
    def __init__(self):
        self.fs = 44100
        self.buffer_size = int(self.fs * 3)
        self.mic_fifo = np.zeros(self.buffer_size)
        self.out_fifo = np.zeros(self.buffer_size)
        self.mixer_queue = queue.Queue()
        self.active_sounds = [] 
        self.lock = threading.Lock()

    def get_source_data(self, source_type):
        with self.lock:
            return self.mic_fifo.copy() if source_type == 'mic' else self.out_fifo.copy()

    def audio_callback(self, indata, outdata, frames, time_info, status):
        with self.lock:
            self.mic_fifo = np.roll(self.mic_fifo, -frames)
            self.mic_fifo[-frames:] = indata[:, 0]
        
        while not self.mixer_queue.empty():
            self.active_sounds.append(self.mixer_queue.get_nowait())

        mixed_out = np.zeros(frames)
        still_playing = []
        
        for sound in self.active_sounds:
            take = min(len(sound), frames)
            mixed_out[:take] += sound[:take]
            if len(sound) > frames:
                still_playing.append(sound[frames:])
        
        self.active_sounds = still_playing
        final_signal = np.clip(mixed_out, -1.0, 1.0)
        outdata[:, 0] = final_signal
        
        with self.lock:
            self.out_fifo = np.roll(self.out_fifo, -frames)
            self.out_fifo[-frames:] = final_signal

    def run(self):
        # ts definition with random ranges instead of fixed numbers
        # Format: [source_type, (min_seconds, max_seconds)]
        ts = [
            ['mic', (2, 7)],
            ['output', (2, 7)],
            ['mic', (2, 7)],
            ['output', (2, 7)]
        ]

        for i, (source, duration_range) in enumerate(ts):
            layer = LayerThread(i+1, source, duration_range, self.fs, self.mixer_queue, self)
            layer.start()

        with sd.Stream(channels=1, samplerate=self.fs, callback=self.audio_callback):
            print(f"--- System Running: Randomized 2-7s Intervals ---")
            while True:
                sd.sleep(1000)

if __name__ == "__main__":
    try:
        MultiLayerProcessor().run()
    except KeyboardInterrupt:
        print("\nExiting...")