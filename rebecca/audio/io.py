import threading
import queue
import sounddevice as sd
import numpy as np

class InputStream:
    def __init__(self, rate=44100, chunk_size=1024):
        self.rate = rate
        self.chunk_size = chunk_size
        self.buffer = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.running = False

    def _run(self):
        # Background loop filling the buffer
        with sd.InputStream(samplerate=self.rate, channels=1, callback=self._callback):
            while self.running:
                sd.sleep(100)

    def _callback(self, indata, frames, time, status):
        self.buffer.put(indata.copy().flatten())

    def start(self):
        self.running = True
        self.thread.start()

class OutputStream:
    def __init__(self, rate=44100):
        self.rate = rate
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.running = False

    def _run(self):
        while self.running:
            data = self.queue.get() # Waits for data to appear
            sd.play(data, self.rate)
            sd.wait()

    def start(self):
        self.running = True
        self.thread.start()

    def feed(self, sample):
        """Accepts a Sample object and puts its raw data into the playback queue."""
        self.queue.put(sample.data)