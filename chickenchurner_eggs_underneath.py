import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import scipy.interpolate as interp
import os
import time
import random
import sys
import psutil 

class ChickenChurner:
    def __init__(self, base_input="chickens.wav", loops=8, fade_decrement=0.25):
        self.base_input = base_input
        self.num_loops = loops
        self.fade_decrement = fade_decrement
        self.ghost_factor = 1.19
        self.fs = 44100  
        
        self.source_audio = None
        self.previous_mix = None # The parent for the next ghost
        self.accumulator = None  # The permanent background history
        self.created_files = [] 

    def _progress_bar(self, current, total, prefix=''):
        percent = float(current) / total
        cpu_usage = psutil.cpu_percent()
        bar_len = 20
        filled_len = int(bar_len * percent)
        bar = '█' * filled_len + '-' * (bar_len - filled_len)
        sys.stdout.write(f'\r{prefix} |{bar}| {percent:.1%} [CPU: {cpu_usage:>4}%]')
        sys.stdout.flush()
        if current >= total: print()

    def get_sound(self):
        choice = input("Press 'M' to record from Mic, or 'F' to use file: ").strip().lower()
        if choice == 'm':
            self.source_audio = self._capture_live_audio(duration=3.0)
        else:
            self.source_audio = self._load_file_audio()
        # Seed the feedback with the first source
        self.previous_mix = self.source_audio
        return self.source_audio

    def _load_file_audio(self):
        if not os.path.exists(self.base_input):
            raise FileNotFoundError(f"File {self.base_input} not found.")
        fs, data = wavfile.read(self.base_input)
        self.fs = fs
        if data.dtype == np.int16: data = data.astype(np.float32) / 32768.0
        return data[:, 0] if len(data.shape) > 1 else data

    def _capture_live_audio(self, duration=3.0):
        print(f"Recording for {duration} seconds...")
        recording = sd.rec(int(duration * self.fs), samplerate=self.fs, channels=1, dtype='float32')
        sd.wait()
        return recording.flatten()

    def transform_slow_down(self, audio_data):
        old_indices = np.arange(len(audio_data))
        new_indices = np.linspace(0, len(audio_data) - 1, int(len(audio_data) * self.ghost_factor))
        interpolant = interp.interp1d(old_indices, audio_data, kind='linear')
        return interpolant(new_indices)

    def apply_curved_fade(self, audio_data, duration):
        if duration <= 0: return audio_data
        fade_samples = min(int(duration * self.fs), len(audio_data))
        curve = np.power(np.linspace(0.0, 1.0, fade_samples), 2)
        output = audio_data.copy()
        output[:fade_samples] *= curve
        return output

    def output(self, audio_data, iteration):
        filename = f"chickens_{iteration:02d}.wav"
        self.created_files.append(filename)
        print(f"\n[Playing {filename}]")
        sd.play(audio_data, self.fs)
        sd.wait()
        wavfile.write(filename, self.fs, audio_data.astype(np.float32))

    def perform(self):
        self.get_sound()
        initial_fade_len = len(self.source_audio) / self.fs

        for i in range(1, self.num_loops + 1):
            print(f"\n{'='*55}\n   LOOP {i} / {self.num_loops}\n{'='*55}")
            
            # 1. GENERATE THE NEW GHOST (Slowing down the PREVIOUS mix)
            new_ghost = self.transform_slow_down(self.previous_mix)
            
            # Stochastic Effects on this new branch
            if random.random() > 0.5:
                print("Effect: Distortion")
                new_ghost = np.clip(new_ghost * 2.5, -1.0, 1.0)
            if random.random() > 0.5:
                print("Effect: Reverb")
                ir = np.random.normal(0, 0.01, int(self.fs * 0.5)) * np.exp(-5 * np.linspace(0, 1, int(self.fs * 0.5)))
                new_ghost = np.convolve(new_ghost, ir, mode='full')

            # 2. SCALE AND FADE THE NEW GHOST ONLY
            scaled_new_ghost = new_ghost * (1.0 / i)
            fade_len = max(0, initial_fade_len - (self.fade_decrement * (i - 1)))
            faded_new_ghost = self.apply_curved_fade(scaled_new_ghost, fade_len)

            # 3. ACCUMULATE THE SHADOWS
            if self.accumulator is None:
                self.accumulator = faded_new_ghost
            else:
                max_len = max(len(self.accumulator), len(faded_new_ghost))
                temp_acc = np.zeros(max_len)
                temp_acc[:len(self.accumulator)] += self.accumulator
                temp_acc[:len(faded_new_ghost)] += faded_new_ghost
                self.accumulator = temp_acc

            # 4. FINAL MIX: Original Source (Locked Speed) + Accumulator (The Melting Shadows)
            total_len = max(len(self.source_audio), len(self.accumulator))
            final_mix = np.zeros(total_len)
            final_mix[:len(self.source_audio)] += self.source_audio
            final_mix[:len(self.accumulator)] += self.accumulator
            
            # Global Normalization
            max_val = np.max(np.abs(final_mix))
            if max_val > 0: final_mix /= max_val

            # Update the seed for the next loop's ghost
            self.previous_mix = final_mix
            
            self.output(final_mix, i)
            time.sleep(0.5)
        
        # Cleanup
        for file in self.created_files[1:]:
            if os.path.exists(file): os.remove(file)

if __name__ == "__main__":
    churner = ChickenChurner()
    churner.perform()