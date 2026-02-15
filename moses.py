import sounddevice as sd
import numpy as np
import threading
import queue
import time
from scipy.signal import butter, lfilter

class LayerThread(threading.Thread):
    def __init__(self, layer_id, source_type, duration, fs, mixer_queue, processor):
        super().__init__(daemon=True)
        self.layer_id = layer_id
        self.source_type = source_type
        self.duration = duration
        self.fs = fs
        self.mixer_queue = mixer_queue
        self.processor = processor
        self.running = True

    def stretch_and_verb(self, data):
        if data.size == 0: 
            return None
        try:
            # Time Stretch +19%
            n_samples = len(data)
            new_indices = np.linspace(0, n_samples - 1, int(n_samples * 1.19))
            stretched = np.interp(new_indices, np.arange(n_samples), data.flatten())
            
            # Reverb
            delay = int(self.fs * 0.15)
            out = np.zeros_like(stretched)
            if len(stretched) > delay:
                out[delay:] = stretched[:-delay] * 0.4
            return ((stretched + out) * 0.5).astype(np.float32)
        except Exception as e:
            return None

    def low_pass(self, data):
        nyquist = 0.5 * self.fs
        b, a = butter(2, 2500 / nyquist, btype='low')
        return lfilter(b, a, data, axis=0)

    def run(self):
        print(f"Layer {self.layer_id} ({self.source_type}) initialized.")
        while self.running:
            captured_chunk = self.processor.get_source_data(self.source_type)
            
            # Only process if we have a full 3s buffer
            if len(captured_chunk) < (self.fs * 3):
                time.sleep(0.5)
                continue

            if self.source_type == 'output':
                captured_chunk = self.low_pass(captured_chunk)
            
            processed = self.stretch_and_verb(captured_chunk)
            
            if processed is not None:
                self.mixer_queue.put(processed)
            
            time.sleep(self.duration)

class MultiLayerProcessor:
    def __init__(self):
        self.fs = 44100
        self.buffer_size = int(self.fs * 3)
        self.mic_fifo = np.zeros(self.buffer_size)
        self.out_fifo = np.zeros(self.buffer_size)
        self.mixer_queue = queue.Queue()
        self.active_layers = []
        self.lock = threading.Lock()

    def get_source_data(self, source_type):
        with self.lock:
            if source_type == 'mic':
                return self.mic_fifo.copy()
            return self.out_fifo.copy()

    def audio_callback(self, indata, outdata, frames, time_info, status):
        # 1. Update Input Buffer (Rolling)
        with self.lock:
            self.mic_fifo = np.roll(self.mic_fifo, -frames)
            self.mic_fifo[-frames:] = indata[:, 0]
        
        # 2. Pull new processed audio from threads
        while not self.mixer_queue.empty():
            self.active_layers.append(self.mixer_queue.get_nowait())

        # 3. Mixing
        mixed_buffer = np.zeros(frames)
        to_remove = []

        for i, data in enumerate(self.active_layers):
            take = min(len(data), frames)
            mixed_buffer[:take] += data[:take]
            self.active_layers[i] = data[take:]
            if len(self.active_layers[i]) == 0:
                to_remove.append(i)

        for i in reversed(to_remove):
            self.active_layers.pop(i)

        # 4. Output + Loopback Recording
        final_out = np.clip(mixed_buffer, -1.0, 1.0)
        outdata[:, 0] = final_out
        
        with self.lock:
            self.out_fifo = np.roll(self.out_fifo, -frames)
            self.out_fifo[-frames:] = final_out

    def run(self, x, y):
        # Start Threads
        l1 = LayerThread(1, 'mic', x, self.fs, self.mixer_queue, self)
        l2 = LayerThread(2, 'output', y, self.fs, self.mixer_queue, self)
        l3 = LayerThread(3, 'mic', x, self.fs, self.mixer_queue, self)
        
        for thread in [l1, l2, l3]: 
            thread.start()

        with sd.Stream(channels=1, samplerate=self.fs, callback=self.audio_callback):
            print("--- System Running ---")
            print(f"Sampling Mic every {x}s and Output every {y}s.")
            while True:
                sd.sleep(1000)

if __name__ == "__main__":
    try:
        MultiLayerProcessor().run(x=2, y=4)
    except KeyboardInterrupt:
        print("\nStopping...")