#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/"src"))
from geas35.experiments.rl.step16_v1_4 import run_step16_v1_4
if __name__=="__main__":print(run_step16_v1_4(project_root=ROOT))
