import sounddevice as sd
import numpy as np
import threading
import time
from scipy.signal import butter, lfilter

# --- Configuration ---
fs = 44100
capture_dur = 2
stretch_factor = 1.19
stagger_delay = 0.5

# Effect Parameters
CUTOFF_FREQ = 2000  # Low-pass filter frequency in Hz
DRIVE = 1.5         # Saturation/Distortion intensity (1.0 = clean, 5.0 = heavy)

class Layer:
    def __init__(self, data):
        self.data = data
        self.ptr = 0
        self.is_active = False

    def get_samples(self, frames):
        if not self.is_active:
            return np.zeros(frames, dtype=np.float32)
        
        n_samples = len(self.data)
        indices = np.arange(self.ptr, self.ptr + frames)
        chunk = self.data[indices % n_samples]
        
        self.ptr += frames
        if self.ptr >= n_samples:
            self.ptr = 0
            self.evolve()
            
        return chunk

    def evolve(self):
        n_old = len(self.data)
        n_new = int(n_old * stretch_factor)
        new_indices = np.linspace(0, n_old - 1, n_new)
        stretched = np.interp(new_indices, np.arange(n_old), self.data)
        
        max_val = np.max(np.abs(stretched))
        if max_val > 0:
            stretched = (stretched / max_val) * 0.2
            
        self.data = stretched.astype(np.float32)

layers = []
lock = threading.Lock()

# Helper for Low Pass Filter
def low_pass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def audio_callback(outdata, frames, time_info, status):
    mixed = np.zeros(frames, dtype=np.float32)
    
    with lock:
        for layer in layers:
            mixed += layer.get_samples(frames)
    
    # --- EFFECT 1: Soft Clipping / Saturation ---
    # We use np.tanh to create a warm distortion/limiting effect
    mixed = np.tanh(mixed * DRIVE)
    
    # --- EFFECT 2: Master Low Pass Filter ---
    # This removes harsh high frequencies
    try:
        mixed = low_pass_filter(mixed, CUTOFF_FREQ, fs).astype(np.float32)
    except:
        pass # Fallback if filter fails on a short frame
    
    # Final Output Clipping (Hard Limit)
    outdata[:, 0] = np.clip(mixed, -1.0, 1.0)

def main():
    global layers
    print(f"--- Phase 1: Capturing 5 Seeds ---")
    for i in range(5):
        print(f"Recording {i+1}/5...")
        rec = sd.rec(int(capture_dur * fs), samplerate=fs, channels=1, blocking=True)
        raw_data = rec.flatten().astype(np.float32)
        if np.max(np.abs(raw_data)) > 0:
            raw_data = (raw_data / np.max(np.abs(raw_data))) * 0.2
        layers.append(Layer(raw_data))

    print("\n--- Phase 2: Unified Stream with Master Effects ---")
    with sd.OutputStream(channels=1, samplerate=fs, callback=audio_callback):
        for i, layer in enumerate(layers):
            print(f"Adding Layer {i+1} to FX chain...")
            with lock:
                layer.is_active = True
            time.sleep(stagger_delay)
        
        while True:
            time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")