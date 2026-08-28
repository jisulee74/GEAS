#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/"src"))
from geas35.experiments.rl.step15_v2 import prepare_step15_v2
if __name__=="__main__":print(prepare_step15_v2(ROOT))
