from audio.io import InputStream, OutputStream
from audio.sample import Sampler
from audio.effects import AudioTransformer

def main():
    # Setup hardware
    mic = InputStream()
    speakers = OutputStream()
    
    # Setup logic
    sampler = Sampler()
    fx = AudioTransformer()

    # Start independent threads
    mic.start()
    speakers.start()

    print("Independent streams active. Sampling 2 seconds at a time...")

    try:
        while True:
            # The Sampler now builds the Sample object for us
            raw_sample = sampler.record_from_stream(mic, duration_sec=2.0)
            
            # Transform and Play
            processed_sample = raw_sample.transform(fx)
            speakers.feed(processed_sample)
            
    except KeyboardInterrupt:
        print("\nStopping...")

if __name__ == "__main__":
    main()