import sounddevice as sd
import numpy as np
import threading
import queue
import time
import random
import librosa

class CaptureLayer(threading.Thread):
    def __init__(self, layer_id, source_type, duration_range, fs, mixer_queue, processor, initial_delay=4):
        super().__init__(daemon=True)
        self.layer_id = layer_id
        self.source_type = source_type
        self.min_dur, self.max_dur = duration_range
        self.fs = fs
        self.mixer_queue = mixer_queue
        self.processor = processor
        self.initial_delay = initial_delay

    def stretch_and_verb(self, data):
        if data.size == 0 or np.max(np.abs(data)) < 0.005: 
            return None
        try:
            stretched = librosa.effects.time_stretch(data.flatten(), rate=0.84)
            delay = int(self.fs * 0.15)
            out = np.zeros_like(stretched)
            if len(stretched) > delay:
                out[delay:] = stretched[:-delay] * 0.4
            
            # Fade out last 2000 samples
            if len(stretched) > 2000:
                stretched[-2000:] *= np.linspace(1., 0., 2000)
                
            return ((stretched + out) * 0.3).astype(np.float32)
        except:
            return None

    def run(self):
        time.sleep(self.initial_delay)
        while True:
            time.sleep(random.uniform(self.min_dur, self.max_dur))
            
            # Check current allowed capacity from processor
            if self.processor.get_writing_layer_count() >= self.processor.allowed_capacity:
                continue
            
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
        self.writing_layers = []  
        self.lock = threading.Lock()
        
        # Capacity Logic
        self.allowed_capacity = 2
        self.num_capture_layers = 0

    def get_writing_layer_count(self):
        with self.lock:
            return len(self.writing_layers)

    def get_source_data(self, source_type):
        with self.lock:
            return self.mic_fifo.copy() if source_type == 'mic' else self.out_fifo.copy()

    def capacity_controller(self):
        """Slowly oscillates allowed_capacity between 2 and 2*Z."""
        target_max = self.num_capture_layers * 2
        direction = 1 # 1 for increasing, -1 for decreasing
        
        while True:
            time.sleep(10) # Change limit every 10 seconds
            
            new_val = self.allowed_capacity + direction
            
            if new_val >= target_max:
                new_val = target_max
                direction = -1
                print(f"\n--- Capacity Peak: {new_val} (Decreasing...) ---")
            elif new_val <= 2:
                new_val = 2
                direction = 1
                print(f"\n--- Capacity Floor: {new_val} (Increasing...) ---")
            else:
                print(f"\n--- Capacity Shift: {new_val} Layers Allowed ---")
                
            self.allowed_capacity = new_val

    def audio_callback(self, indata, outdata, frames, time_info, status):
        with self.lock:
            self.mic_fifo = np.roll(self.mic_fifo, -frames)
            self.mic_fifo[-frames:] = indata[:, 0]
        
        while not self.mixer_queue.empty():
            with self.lock:
                if len(self.writing_layers) < self.allowed_capacity:
                    self.writing_layers.append(self.mixer_queue.get_nowait())
                else:
                    try: self.mixer_queue.get_nowait()
                    except: pass

        mixed_out = np.zeros(frames)
        still_writing = []
        with self.lock:
            for sound in self.writing_layers:
                take = min(len(sound), frames)
                mixed_out[:take] += sound[:take]
                if len(sound) > frames:
                    still_playing = sound[frames:]
                    still_writing.append(still_playing)
            self.writing_layers = still_writing

        final_signal = np.clip(mixed_out, -1.0, 1.0)
        outdata[:, 0] = final_signal
        
        with self.lock:
            self.out_fifo = np.roll(self.out_fifo, -frames)
            self.out_fifo[-frames:] = final_signal

    def run(self):
        self.num_capture_layers = random.randint(3, 6)
        print(f"--- {self.num_capture_layers} Capture Layers Spawned ---")
        
        # Start Capacity Controller
        threading.Thread(target=self.capacity_controller, daemon=True).start()

        for i in range(self.num_capture_layers):
            source = 'mic' if i % 2 == 0 else 'output'
            CaptureLayer(i+1, source, (2, 7), self.fs, self.mixer_queue, self).start()

        with sd.Stream(channels=1, samplerate=self.fs, callback=self.audio_callback):
            while True:
                sd.sleep(1000)

if __name__ == "__main__":
    try:
        MultiLayerProcessor().run()
    except KeyboardInterrupt:
        print("\nExit.")