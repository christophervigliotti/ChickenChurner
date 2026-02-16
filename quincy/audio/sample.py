class Sample:
    def __init__(self, data, rate):
        self.data = data  # Usually a NumPy array
        self.rate = rate

    def transform(self, transformer):
        return transformer.process(self)