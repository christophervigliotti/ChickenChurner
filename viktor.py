import sounddevice as sd
import numpy as np
import threading
import time

# --- Configuration ---
fs = 44100
capture_dur = 2
stretch_factor = 1.19
stagger_delay = 0.5

class Layer:
    def __init__(self, data):
        self.data = data
        self.ptr = 0
        self.is_active = False

    def get_samples(self, frames):
        if not self.is_active:
            return np.zeros(frames, dtype=np.float32)
        
        n_samples = len(self.data)
        # Generate indices for the current chunk
        indices = np.arange(self.ptr, self.ptr + frames)
        chunk = self.data[indices % n_samples]
        
        # Check if we finished a full loop
        self.ptr += frames
        if self.ptr >= n_samples:
            self.ptr = 0
            self.evolve()
            
        return chunk

    def evolve(self):
        # Time stretch by 19%
        n_old = len(self.data)
        n_new = int(n_old * stretch_factor)
        new_indices = np.linspace(0, n_old - 1, n_new)
        
        stretched = np.interp(new_indices, np.arange(n_old), self.data)
        
        # Normalize to keep layering balanced
        max_val = np.max(np.abs(stretched))
        if max_val > 0:
            stretched = (stretched / max_val) * 0.2
            
        self.data = stretched.astype(np.float32)

# Global list of layer objects
layers = []
lock = threading.Lock()

def audio_callback(outdata, frames, time_info, status):
    # Create a silent canvas for mixing
    mixed = np.zeros(frames, dtype=np.float32)
    
    with lock:
        for layer in layers:
            mixed += layer.get_samples(frames)
    
    # Send the final mix to the single output stream
    outdata[:, 0] = np.clip(mixed, -1.0, 1.0)

def main():
    global layers
    
    # 1. Automatic Capture of 5 Seeds
    print(f"--- Phase 1: Capturing 5 Seeds ({capture_dur}s each) ---")
    for i in range(5):
        print(f"Recording Clip {i+1}/5...")
        rec = sd.rec(int(capture_dur * fs), samplerate=fs, channels=1, blocking=True)
        raw_data = rec.flatten().astype(np.float32)
        
        # Initial normalization
        if np.max(np.abs(raw_data)) > 0:
            raw_data = (raw_data / np.max(np.abs(raw_data))) * 0.2
            
        layers.append(Layer(raw_data))

    # 2. Single Output Stream
    print("\n--- Phase 2: Running Unified Output Stream ---")
    with sd.OutputStream(channels=1, samplerate=fs, callback=audio_callback):
        for i, layer in enumerate(layers):
            print(f"Activating Layer {i+1}...")
            with lock:
                layer.is_active = True
            # Staggered entry into the mix
            time.sleep(stagger_delay)
            
        print("All layers active. Droning indefinitely.")
        while True:
            time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")