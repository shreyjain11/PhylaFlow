import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(
    "/home/yektefai/PhylaFlow/analysis/full_sanity_fixedpair_20260325/analyze_first_last_replay_velocity_loss.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_first_last_replay_velocity_loss",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AnalyzeFirstLastReplayVelocityLossTest(unittest.TestCase):
    def test_select_first_last_velocity_samples_dedupes_and_keeps_ends(self):
        module = _load_module()
        samples = [
            {"newick_tree": "A", "timepoint": 0.0},
            {"newick_tree": "A", "timepoint": 0.0},
            {"newick_tree": "B", "timepoint": 0.1},
            {"newick_tree": "C", "timepoint": 0.2},
        ]
        selected = module._select_first_last_velocity_samples(samples)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]["newick_tree"], "A")
        self.assertEqual(selected[1]["newick_tree"], "C")

    def test_select_first_last_velocity_samples_handles_single_state(self):
        module = _load_module()
        samples = [
            {"newick_tree": "A", "timepoint": 0.0},
            {"newick_tree": "A", "timepoint": 0.0},
        ]
        selected = module._select_first_last_velocity_samples(samples)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["newick_tree"], "A")


if __name__ == "__main__":
    unittest.main()
