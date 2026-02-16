import numpy as np

class Sample:
    def __init__(self, data, rate, name="Unnamed"):
        self.data = data
        self.rate = rate
        self.name = name

    def transform(self, transformer):
        new_data = transformer.process(self.data, self.rate)
        return Sample(new_data, self.rate, name=f"{self.name}_fx")

class Sampler:
    def __init__(self, name, stream):
        self.name = name
        # Get a dedicated pipe for this sampler
        self.queue = stream.get_subscription()
        self.rate = stream.rate
        self.chunk_size = stream.chunk_size

    def record_from_stream(self, duration_sec):
        num_chunks = int((duration_sec * self.rate) / self.chunk_size)
        recorded_data = []
        
        for _ in range(num_chunks):
            recorded_data.append(self.queue.get())
            
        return Sample(np.concatenate(recorded_data), self.rate, name=self.name)