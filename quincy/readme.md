audio_project/
│
├── main.py              # Entry point (orchestrates the stream)
├── audio/
│   ├── __init__.py      # Makes 'audio' a package
│   ├── io.py            # InputStream and OutputStream classes
│   ├── sample.py        # The Sample data container
│   └── effects.py       # AudioTransformer and DSP sub-methods