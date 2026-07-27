import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MDP_CHECKLIST = ROOT / "docs" / "MDP v1 환경 설정 체크리스트 문서 작성 계획.md"
INTEGRATION_TODO = ROOT / "docs" / "integration_todo.md"


class TransitionDocumentationTest(unittest.TestCase):
    def test_transition_docs_exist_and_reference_current_implementation(self):
        checklist = MDP_CHECKLIST.read_text(encoding="utf-8")
        todo = INTEGRATION_TODO.read_text(encoding="utf-8")

        self.assertIn("dynamic next observation subset", checklist)
        self.assertIn("selected_transition_model.json", checklist)
        self.assertIn("PolicyActionProvider", checklist)
        self.assertIn("Implemented Transition Module Status", todo)
        self.assertIn("predict_next_observation", todo)

    def test_documented_transition_paths_exist(self):
        documented_paths = [
            ROOT / "src" / "geas35" / "models" / "transition",
            ROOT / "src" / "geas35" / "models" / "transition" / "inference.py",
            ROOT / "src" / "geas35" / "models" / "transition" / "rollout.py",
            ROOT / "configs" / "experiments" / "transition" / "default.yaml",
        ]

        for path in documented_paths:
            with self.subTest(path=path):
                self.assertTrue(path.exists(), f"Missing documented path: {path}")


if __name__ == "__main__":
    unittest.main()
