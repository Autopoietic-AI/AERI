"""Compatibility entry for existing GeoIG launchers; see aeri.py."""
from pathlib import Path
import runpy,sys
folder=Path(__file__).resolve().parents[2]/"stagefusion"
sys.path.insert(0,str(folder))
if __name__=="__main__":
    runpy.run_path(str(folder/"main_cat101_a5_tig_stagefusion.py"),run_name="__main__")
