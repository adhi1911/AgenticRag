import sys
import os
from pathlib import Path


project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


os.environ['PYTHONPATH'] = str(project_root)


import subprocess

if __name__ == "__main__":

    cmd = [sys.executable, "-m", "streamlit", "run", "src/ui/streamlit_app.py"]
    subprocess.run(cmd)