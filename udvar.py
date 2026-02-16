import sounddevice as sd
import numpy as np
import time

def auto_layered_loop():
    fs = 44100
    capture_dur = 5
    stretch_factor = 1.19
    stagger_delay = 2.5
    
    clips = []
    
    # --- PHASE 1: AUTOMATIC RECORDING ---
    print("--- Phase 1: Capturing 5 Seeds (No stopping) ---")
    for i in range(5):
        print(f"Recording Clip {i+1}/5... (5 seconds)")
        # blocking=True ensures we finish one 5s recording before starting the next
        rec = sd.rec(int(capture_dur * fs), samplerate=fs, channels=1, blocking=True)
        
        # Prepare the clip: float32 and normalized volume for layering
        clip = rec.flatten().astype(np.float32)
        if np.max(np.abs(clip)) > 0:
            clip = (clip / np.max(np.abs(clip))) * 0.3
        clips.append(clip)

    # --- PHASE 2: EVOLVING LAYERS ---
    print("\n--- Phase 2: Starting Staggered Playback ---")
    
    iteration = 1
    try:
        while True:
            print(f"\n--- Cycle {iteration} ---")
            
            for i in range(5):
                # 1. Slow down the current clip by 19%
                n_samples = len(clips[i])
                new_n_samples = int(n_samples * stretch_factor)
                new_indices = np.linspace(0, n_samples - 1, new_n_samples)
                clips[i] = np.interp(new_indices, np.arange(n_samples), clips[i]).astype(np.float32)
                
                # 2. Trigger playback (Non-blocking)
                # This layers the sound over whatever is currently playing
                sd.play(clips[i], fs)
                
                print(f"Voice {i+1} playing: {len(clips[i])/fs:.2f}s")
                
                # 3. Wait 2.5 seconds before starting the next voice
                if i < 4:
                    time.sleep(stagger_delay)
            
            # 4. Wait for the final/longest clip of this cycle to finish
            # before we start the next round of transformations.
            sd.wait()
            iteration += 1

    except KeyboardInterrupt:
        print("\nStopping...")
        sd.stop()

if __name__ == "__main__":
    try:
        auto_layered_loop()
    except Exception as e:
        print(f"An error occurred: {e}")