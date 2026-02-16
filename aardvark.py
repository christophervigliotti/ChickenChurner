import sounddevice as sd
import numpy as np
import threading
import time

# --- Configuration ---
fs = 44100
capture_dur = 2        # Duration of each new seed
grand_loop_dur = 15    # Interval to capture 5 new seeds + resample output
stretch_factor = 1.19
stagger_delay = 0.5

class Layer:
    def __init__(self, data, volume=0.2):
        self.data = data.flatten().astype(np.float32)
        self.ptr = 0
        self.is_active = True
        self.volume = volume

    def get_samples(self, frames):
        if not self.is_active:
            return np.zeros(frames, dtype=np.float32)
        
        n_samples = len(self.data)
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
    mixed = np.zeros(frames, dtype=np.float32)
    with lock:
        for layer in layers:
            mixed += layer.get_samples(frames)
    
    final_signal = np.tanh(mixed * 1.2)
    final_signal = np.clip(final_signal, -1.0, 1.0)
    outdata[:, 0] = final_signal
    
    # Track the output history for the Grand Loop resampling
    master_history.append(final_signal.copy())

def grand_loop_processor():
    global master_history, layers
    
    while True:
        # Wait for the next 15-second cycle
        time.sleep(grand_loop_dur)
        
        print(f"\n--- [Cycle Triggered] Resampling Output & Harvesting New Seeds ---")
        
        # 1. Resample the Master Output (The "Grand Loop")
        with lock:
            if master_history:
                recorded_mix = np.concatenate(master_history)
                master_history = [] # Reset history
                
                target_samples = int(grand_loop_dur * fs)
                if len(recorded_mix) > target_samples:
                    recorded_mix = recorded_mix[-target_samples:]
                
                # Add the master resample as a low-volume foundation layer
                new_grand = Layer(recorded_mix, volume=0.1)
                layers.append(new_grand)

        # 2. Capture 5 New Seeds from Mic (Background)
        # We do this in a sub-loop so the main Grand Loop can eventually restart
        for i in range(5):
            print(f"   > Harvesting Mic Seed {i+1}/5...")
            # Use blocking=True here because we are in a background thread
            new_rec = sd.rec(int(capture_dur * fs), samplerate=fs, channels=1, blocking=True)
            
            new_seed = Layer(new_rec, volume=0.15)
            with lock:
                layers.append(new_seed)
                # Cleanup: Prevent the list from growing infinitely (Keep last 25 layers)
                if len(layers) > 25:
                    layers.pop(0)
            
            time.sleep(stagger_delay)

def main():
    global layers
    
    # Initial Start: Capture first 5 seeds
    print("--- Phase 1: Initial Seed Capture (10 seconds) ---")
    for i in range(5):
        print(f"Recording Initial Seed {i+1}/5...")
        rec = sd.rec(int(capture_dur * fs), samplerate=fs, channels=1, blocking=True)
        layers.append(Layer(rec, volume=0.2))

    # Start the Engine
    with sd.OutputStream(channels=1, samplerate=fs, callback=audio_callback):
        # Kick off the periodic harvester/resampler
        threading.Thread(target=grand_loop_processor, daemon=True).start()
        
        print("\n--- System Operational: Press Ctrl+C to Stop ---")
        while True:
            time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")