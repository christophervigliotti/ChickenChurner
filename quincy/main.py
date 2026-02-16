import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio.io import InputStream
from audio.effects import AudioTransformer

from audio.io import InputStream, OutputStream
from audio.effects import AudioTransformer

def main():
    mic = InputStream()
    speakers = OutputStream()
    fx = AudioTransformer()

    print("Processing... Press Ctrl+C to stop.")
    
    try:
        while True:
            # 1. Capture
            raw_sample = mic.capture()
            
            # 2. Transform
            processed_sample = raw_sample.transform(fx)
            
            # 3. Output
            speakers.play(processed_sample)
            
    except KeyboardInterrupt:
        print("\nStream stopped.")

if __name__ == "__main__":
    main()