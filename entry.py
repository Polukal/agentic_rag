import os
import subprocess
import sys

def run_streamlit():
    file_path = os.path.join(os.path.dirname(__file__), "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", file_path])

if __name__ == "__main__":
    run_streamlit()
