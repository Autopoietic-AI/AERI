"""AERI command-line entry; forwards arguments to the unchanged training script."""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    folder=Path(__file__).resolve().parent/"stagefusion"
    sys.path.insert(0,str(folder))
    runpy.run_path(str(folder/"main_cat101_a5_tig_stagefusion.py"),run_name="__main__")
