import numpy as np

class Sample:
    def __init__(self, data, rate):
        self.data = data
        self.rate = rate

    def transform(self, transformer):
        # Passes raw data to the transformer, gets raw data back
        new_data = transformer.process(self.data, self.rate)
        # Wraps the result in a new Sample object
        return Sample(new_data, self.rate)

class Sampler:
    def record_from_stream(self, stream, duration_sec):
        """Creates a Sample by pulling chunks from an active InputStream buffer."""
        # Calculate how many chunks we need to grab to hit the duration
        num_chunks = int((duration_sec * stream.rate) / stream.chunk_size)
        recorded_data = []
        
        for _ in range(num_chunks):
            # This pulls from the queue.Queue() in InputStream
            chunk = stream.buffer.get() 
            recorded_data.append(chunk)
            
        if not recorded_data:
            return Sample(np.array([]), stream.rate)

        flat_data = np.concatenate(recorded_data)
        return Sample(flat_data, stream.rate)

    def load_from_file(self, file_path):
        """Placeholder for loading a file."""
        print(f"Loading {file_path}...")
        # For now, returns 1 second of silence
        return Sample(np.zeros(44100), 44100)