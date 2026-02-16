import sounddevice as sd
import numpy as np
import threading
import time

# --- Configuration ---
fs = 44100
capture_dur = 1        
grand_loop_dur = .5    
stretch_factor = 1.19
stagger_delay = .4

class Layer:
    def __init__(self, data, volume=0.2):
        self.data = data.flatten().astype(np.float32)
        self.ptr = 0
        self.is_active = False
        self.volume = volume

    def get_samples(self, frames):
        if not self.is_active:
            return np.zeros(frames, dtype=np.float32)
        
        n_samples = len(self.data)
        # Use modulo to wrap around the buffer
        indices = (np.arange(self.ptr, self.ptr + frames)) % n_samples
        chunk = self.data[indices]
        
        self.ptr += frames
        if self.ptr >= n_samples:
            self.ptr = 0
            self.evolve()
            
        return chunk * self.volume

    def evolve(self):
        n_old = len(self.data)
        n_new = int(n_old * stretch_factor)
        new_indices = np.linspace(0, n_old - 1, n_new)
        stretched = np.interp(new_indices, np.arange(n_old), self.data)
        
        max_val = np.max(np.abs(stretched))
        if max_val > 0:
            stretched = stretched / max_val
            
        self.data = stretched.astype(np.float32)

# --- Global State ---
layers = []
master_history = [] 
lock = threading.Lock()

def audio_callback(outdata, frames, time_info, status):
    # Standard output callback signature
    mixed = np.zeros(frames, dtype=np.float32)
    
    with lock:
        for layer in layers:
            # Basic additive mixing
            mixed += layer.get_samples(frames)
    
    # Saturation and Clipping
    final_signal = np.tanh(mixed * 1.2)
    final_signal = np.clip(final_signal, -1.0, 1.0)
    
    # Send to the hardware output (ensuring correct shape)
    outdata[:, 0] = final_signal
    
    # Store for the Grand Loop
    master_history.append(final_signal.copy())

def grand_loop_processor():
    global master_history, layers
    print(f"--- Grand Loop Active: Sampling every {grand_loop_dur}s ---")
    
    while True:
        time.sleep(grand_loop_dur)
        
        with lock:
            if len(master_history) == 0:
                continue
            # Combine all chunks collected during the sleep period
            recorded_mix = np.concatenate(master_history)
            master_history = [] 
        
        # Keep only the last 15 seconds worth of samples
        target_samples = int(grand_loop_dur * fs)
        if len(recorded_mix) > target_samples:
            recorded_mix = recorded_mix[-target_samples:]
            
        print(f"\n[Grand Loop] Resampling and adding new master layer...")
        
        # Create a new evolving layer from the output we just heard
        new_grand_layer = Layer(recorded_mix, volume=0.15)
        new_grand_layer.is_active = True
        
        with lock:
            layers.append(new_grand_layer)
            # Prevent memory explosion: keep 5 seeds + last 5 grand loops
            if len(layers) > 10:
                layers.pop(5) 

def main():
    global layers
    
    # 1. Capture 5 Seeds
    print(f"--- Phase 1: Capturing 5 Seeds (2s each) ---")
    for i in range(5):
        print(f"Recording Seed {i+1}/5...")
        rec = sd.rec(int(capture_dur * fs), samplerate=fs, channels=1, blocking=True)
        layers.append(Layer(rec, volume=0.2))

    # 2. Open Output Stream
    # Note: Using OutputStream to avoid complex multi-input/output logic
    with sd.OutputStream(channels=1, samplerate=fs, callback=audio_callback):
        # Staggered activation
        for layer in layers:
            with lock:
                layer.is_active = True
            time.sleep(stagger_delay)
            
        # 3. Start the background sampler
        threading.Thread(target=grand_loop_processor, daemon=True).start()
        
        print("\n--- Audio Engine Running ---")
        while True:
            time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")