from audio.io import InputStream, OutputStream
from audio.sample import Sampler
from audio.effects import AudioTransformer
import threading
import time

def run_sampler_loop(sampler, speakers, fx, duration, delay):
    print(f"-> {sampler.name} loop running.")
    while True:
        raw_sample = sampler.record_from_stream(duration_sec=duration)
        processed_sample = raw_sample.transform(fx)
        speakers.feed(processed_sample, delay_sec=delay)

def main():
    mic = InputStream()
    speakers = OutputStream()
    fx = AudioTransformer()

    mic.start()
    speakers.start()

    configs = [
        {"name": "2s-Loop", "dur": 2.0, "delay": 2.0},
        {"name": "8s-Loop", "dur": 8.0, "delay": 2.0},
        {"name": "5s-Loop", "dur": 5.0, "delay": 2.0}
    ]

    for conf in configs:
        # Each sampler now gets its own 'subscription' to the mic
        s = Sampler(name=conf["name"], stream=mic)
        t = threading.Thread(
            target=run_sampler_loop, 
            args=(s, speakers, fx, conf["dur"], conf["delay"]),
            daemon=True
        )
        t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExit.")

if __name__ == "__main__":
    main()