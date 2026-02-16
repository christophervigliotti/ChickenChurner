import threading
import queue
import sounddevice as sd
import numpy as np

audio_lock = threading.Lock()

class InputStream:
    def __init__(self, rate=44100, chunk_size=1024):
        self.rate = rate
        self.chunk_size = chunk_size
        self.subscribers = [] # List of queues for each sampler
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.running = False

    def get_subscription(self):
        """Creates and returns a new queue for a specific sampler."""
        q = queue.Queue()
        self.subscribers.append(q)
        return q

    def _run(self):
        with sd.InputStream(samplerate=self.rate, channels=1, callback=self._callback):
            while self.running:
                sd.sleep(100)

    def _callback(self, indata, frames, time, status):
        flat_data = indata.copy().flatten()
        # Send a copy of the audio to every sampler's queue
        for q in self.subscribers:
            q.put(flat_data)

    def start(self):
        self.running = True
        self.thread.start()

class OutputStream:
    def __init__(self, rate=44100):
        self.rate = rate

    def start(self):
        print("Output Stream Ready.")

    def _play_task(self, data):
        with audio_lock:
            sd.play(data, self.rate)
            sd.wait()

    def feed(self, sample, delay_sec=0.0):
        if delay_sec <= 0:
            threading.Thread(target=self._play_task, args=(sample.data,), daemon=True).start()
        else:
            threading.Timer(delay_sec, self._play_task, args=(sample.data,)).start()