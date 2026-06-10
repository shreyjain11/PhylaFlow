import math
import os
import re
import shutil
import unittest

from utils.metric_utils import loglh_for_tree_list


_TRANSLATE_LINE_RE = re.compile(r"^\s*(\d+)\s+([^,;]+)")


def _strip_leading_nexus_annotations(newick: str) -> str:
    cleaned = newick.strip()
    while cleaned.startswith("["):
        end = cleaned.find("]")
        if end == -1:
            break
        cleaned = cleaned[end + 1 :].lstrip()
    return cleaned


def _apply_translation(newick: str, translation: dict) -> str:
    if not translation:
        return newick

    def repl(match):
        prefix, label = match.group(1), match.group(2)
        return f"{prefix}{translation.get(label, label)}:"

    return re.sub(r"([,(])\s*(\d+)\s*:", repl, newick)


def _read_trees_from_tprobs(path: str, max_trees: int) -> list:
    translation = {}
    trees = []
    in_translate = False
    in_tree = False
    tree_buf = []

    with open(path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            lower = line.lower()

            if lower.startswith("translate"):
                in_translate = True
                continue

            if in_translate:
                if ";" in line:
                    in_translate = False
                match = _TRANSLATE_LINE_RE.match(line.rstrip(",;"))
                if match:
                    translation[match.group(1)] = match.group(2)
                continue

            if lower.startswith("tree "):
                in_tree = True
                after_eq = line.split("=", 1)[-1]
                tree_buf.append(after_eq)
                if ";" in line:
                    in_tree = False
            elif in_tree:
                tree_buf.append(line)

            if not in_tree and tree_buf:
                tree_str = " ".join(tree_buf)
                tree_buf = []
                tree_str = _strip_leading_nexus_annotations(tree_str)
                tree_str = tree_str.strip().rstrip(";")
                tree_str = _apply_translation(tree_str, translation)
                trees.append(tree_str)
                if len(trees) >= max_trees:
                    break

    return trees


class TestLogLikelihoodForTreeList(unittest.TestCase):
    def test_loglh_for_tree_list_from_tprobs(self):
        msa_path = os.environ.get("PHYLAFLOW_MSA_PATH")
        tprobs_path = os.environ.get("PHYLAFLOW_TPROBS_PATH")
        if not msa_path or not tprobs_path:
            self.skipTest(
                "Set PHYLAFLOW_MSA_PATH and PHYLAFLOW_TPROBS_PATH to run this test."
            )
        if not shutil.which("raxml-ng"):
            self.skipTest("raxml-ng not found on PATH.")

        max_trees = int(os.environ.get("PHYLAFLOW_LOGLH_MAX_TREES", "2"))
        trees = _read_trees_from_tprobs(tprobs_path, max_trees=max_trees)
        if not trees:
            self.fail("No trees found in the provided tprobs file.")

        loglhs = loglh_for_tree_list(msa_path, trees, threads=1)
        self.assertEqual(len(loglhs), len(trees))
        for value in loglhs:
            self.assertIsInstance(value, float)
            self.assertFalse(math.isnan(value))


if __name__ == "__main__":
    unittest.main()
