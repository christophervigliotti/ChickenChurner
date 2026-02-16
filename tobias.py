import sounddevice as sd
import numpy as np
import threading
import time

# --- Global State ---
fs = 44100
step_duration = 5  # Length of each new mic capture
stretch_factor = 1.19
loop_buffer = np.zeros(int(step_duration * fs), dtype=np.float32)
buffer_lock = threading.Lock()
current_ptr = 0

def audio_callback(outdata, frames, time_info, status):
    global current_ptr
    with buffer_lock:
        n_samples = len(loop_buffer)
        if n_samples == 0:
            outdata.fill(0)
            return
        indices = (np.arange(current_ptr, current_ptr + frames)) % n_samples
        outdata[:, 0] = loop_buffer[indices]
        current_ptr = (current_ptr + frames) % n_samples

def processor_thread():
    global loop_buffer
    
    print("--- System Initializing ---")
    
    # 1. Initial Seed Capture
    print(f"Initial capture: Speak now for {step_duration}s...")
    recording = sd.rec(int(step_duration * fs), samplerate=fs, channels=1, blocking=True)
    with buffer_lock:
        loop_buffer = recording.flatten().astype(np.float32)

    iteration = 1
    while True:
        # 2. Record NEW audio while the old loop continues to play
        print(f"\n[Iteration {iteration}] Recording new layer...")
        new_mic_data = sd.rec(int(step_duration * fs), samplerate=fs, channels=1, blocking=True)
        new_mic_data = new_mic_data.flatten().astype(np.float32)

        with buffer_lock:
            # 3. Stretch the EXISTING loop
            old_data = loop_buffer
            n_old = len(old_data)
            n_new = int(n_old * stretch_factor)
            
            new_indices = np.linspace(0, n_old - 1, n_new)
            stretched_base = np.interp(new_indices, np.arange(n_old), old_data)
            
            # 4. Fold the NEW mic recording into the start of the stretched buffer
            # We use a 50/50 mix for the overlap area
            n_mic = len(new_mic_data)
            combined = stretched_base.copy()
            
            # Mix new mic data into the beginning of the expanded loop
            combined[:n_mic] = (combined[:n_mic] * 0.6) + (new_mic_data * 0.4)
            
            # 5. Normalize to prevent buildup distortion
            max_val = np.max(np.abs(combined))
            if max_val > 0.01:
                combined = (combined / max_val) * 0.8
            
            loop_buffer = combined
            
            print(f"Merged. New Loop Length: {len(loop_buffer)/fs:.2f}s")
            iteration += 1

if __name__ == "__main__":
    # Start the Output Stream (Non-blocking)
    stream = sd.OutputStream(channels=1, samplerate=fs, callback=audio_callback)
    
    try:
        with stream:
            # Start the background recorder/processor
            proc = threading.Thread(target=processor_thread, daemon=True)
            proc.start()
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")