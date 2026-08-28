#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/"src"))
from geas35.experiments.rl.v2_readiness import build_v2_readiness_manifest
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--test-summary",required=True);args=parser.parse_args();print(build_v2_readiness_manifest(ROOT,args.test_summary))
