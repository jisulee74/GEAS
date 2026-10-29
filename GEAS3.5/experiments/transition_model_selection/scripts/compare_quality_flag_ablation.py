"""Build paired quality-flag ablation reports after all v1.2 runs finish."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))
from geas35.models.transition.quality_flag_ablation import compare_variant_roots
BASE=ROOT/"experiments/transition_model_selection/artifacts/step12_v1_2"
compare_variant_roots(BASE/"deduplicated_with_quality_flags", BASE/"deduplicated_without_quality_flags", BASE/"ablation_comparison", dataset_root=ROOT/"offline_dataset_preparation/datasets/5_rl_dataset")
