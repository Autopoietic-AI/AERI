"""Compatibility import; implementation is in stagefusion/model_a5_tig_stagefusion.py."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from stagefusion.model_a5_tig_stagefusion import *  # noqa: F401,F403
