"""Download and prepare the Tiny Shakespeare dataset."""
import os
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(DATA_DIR, "input.txt")

if __name__ == "__main__":
    if os.path.exists(OUTPUT_PATH):
        print(f"Dataset already exists at {OUTPUT_PATH}")
    else:
        print(f"Downloading Tiny Shakespeare to {OUTPUT_PATH}...")
        urllib.request.urlretrieve(DATA_URL, OUTPUT_PATH)
        print(f"Done. {os.path.getsize(OUTPUT_PATH):,} bytes")
