import unittest
import os

from utils.metric_utils import (
    kl_divergence_topological_distributions,
    kl_divergence_tree_topology_distributions,
    topk_posterior_tree_recall,
    split_bipartition_frequency_correlation,
    raxmlng_loglh_batch,
    compare_likelihood_distributions,
    compare_branch_length_distributions
)
from utils.random_tree import Tree
from data.dataset import TreeDataset
import random
from utils.utils import number_to_name_newick


class TestMetricUtils(unittest.TestCase):
    def setUp(self):
        self.tree_a = str(Tree(num_leaves=50, random=True))
        self.tree_b = str(Tree(num_leaves=50, random=True))
        self.tree_c = str(Tree(num_leaves=50, random=True))    
        self.num_leaves = 50

    def test_kl_divergence_matches_identity(self):
        posterior = [self.tree_a, self.tree_a, self.tree_b]
        sampled_same = list(posterior)
        sampled_diff = [self.tree_c, self.tree_c, self.tree_c]

        kl_same = kl_divergence_topological_distributions(
            posterior, sampled_same, num_leaves=self.num_leaves
        )['kl_divergence_topological']
        kl_diff = kl_divergence_topological_distributions(
            posterior, sampled_diff, num_leaves=self.num_leaves
        )['kl_divergence_topological']

        self.assertLess(kl_same, 1e-8)
        self.assertGreater(kl_diff, 1e-4)

    def test_tree_topology_kl_matches_identity(self):
        posterior = [self.tree_a, self.tree_a, self.tree_b]
        sampled_same = list(posterior)
        sampled_diff = [self.tree_c, self.tree_c, self.tree_c]

        kl_same = kl_divergence_tree_topology_distributions(
            posterior, sampled_same
        )["kl_divergence_tree_topology"]
        kl_diff = kl_divergence_tree_topology_distributions(
            posterior, sampled_diff
        )["kl_divergence_tree_topology"]

        self.assertLess(kl_same, 1e-8)
        self.assertGreater(kl_diff, 1e-4)

    def test_topk_posterior_tree_recall(self):
        posterior = [
            self.tree_a,
            self.tree_a,
            self.tree_a,
            self.tree_b,
            self.tree_b,
            self.tree_c,
        ]
        sampled = [self.tree_a, self.tree_b, str(Tree(num_leaves=50, random=True))]

        metrics = topk_posterior_tree_recall(posterior, sampled, top_ks=(1, 2, 3))

        self.assertEqual(metrics["posterior_topology_recall_at_1"], 1.0)
        self.assertEqual(metrics["posterior_topology_recall_at_2"], 1.0)
        self.assertAlmostEqual(metrics["posterior_topology_recall_at_3"], 2.0 / 3.0)
        self.assertEqual(metrics["posterior_topology_mass_recall_at_1"], 1.0)
        self.assertEqual(metrics["posterior_topology_mass_recall_at_2"], 1.0)
        self.assertAlmostEqual(
            metrics["posterior_topology_mass_recall_at_3"],
            5.0 / 6.0,
        )

    def test_bipartition_frequency_correlation(self):
        posterior = [str(Tree(num_leaves=self.num_leaves, random=True)) for _ in range(5)]
        sampled_same = list(posterior)
        sampled_diff = [str(Tree(num_leaves=self.num_leaves, random=True)) for _ in range(5)]

        # corr_same = split_bipartition_frequency_correlation(
        #     posterior, sampled_same, num_leaves=self.num_leaves
        # )
        corr_diff = split_bipartition_frequency_correlation(
            posterior, sampled_diff, num_leaves=self.num_leaves
        )['bipartition_frequency_correlation']

        # self.assertAlmostEqual(corr_same, 1.0, places=6)
        self.assertLess(corr_diff, 0.5)
    
    def test_likelihood_calculation(self):
        data_root = os.environ.get("PHYLAFLOW_EXAMPLE_DATA_ROOT")
        if not data_root:
            self.skipTest("set PHYLAFLOW_EXAMPLE_DATA_ROOT to run likelihood fixture test")
        nexus_root = os.path.join(data_root, "nexus")
        mrbayes_root = os.path.join(data_root, "runs")

        dataset = TreeDataset(
            nexus_root=nexus_root,
            mrbayes_root=mrbayes_root)

        posterior_trees = dataset.return_posterior_trees(0)
        if len(posterior_trees) > 1000:
            posterior_trees = random.sample(posterior_trees, 1000)
        num_leaves = dataset.return_number_leaves(0)
        sampled = [str(Tree(num_leaves=num_leaves, random=True)) for _ in range(len(posterior_trees))]
        mapping = dataset.return_nexus_number_to_name(0)
        sampled = [number_to_name_newick(i, mapping, True) for i in sampled]
        posterior_trees = [number_to_name_newick(i, mapping, False) for i in posterior_trees]

        real_posterior_log = raxmlng_loglh_batch(
            nexus_path=dataset.return_nexus_filepath(0),
            newicks=posterior_trees,
            model="JC",
            threads=1
        )

        random_sampled_log = raxmlng_loglh_batch(
            nexus_path=dataset.return_nexus_filepath(0),
            newicks=sampled,
            model="JC",
            threads=1
        )

        self.assertGreater(
            sum(real_posterior_log)/len(real_posterior_log),
            sum(random_sampled_log)/len(random_sampled_log)
        )

        result = compare_likelihood_distributions(dataset.return_nexus_filepath(0), true_trees=posterior_trees, sampled_trees=sampled, threads=1)
        self.assertGreater(result['avg_true_loglh'], result['avg_sampled_loglh'])
        self.assertGreater(result['diff_avg_loglh'], 0)
    
    def test_compare_branchlength_distribution(self):
        data_root = os.environ.get("PHYLAFLOW_EXAMPLE_DATA_ROOT")
        if not data_root:
            self.skipTest("set PHYLAFLOW_EXAMPLE_DATA_ROOT to run branch-length fixture test")
        nexus_root = os.path.join(data_root, "nexus")
        mrbayes_root = os.path.join(data_root, "runs")

        dataset = TreeDataset(
            nexus_root=nexus_root,
            mrbayes_root=mrbayes_root)

        posterior_trees = dataset.return_posterior_trees(0)
        if len(posterior_trees) > 1000:
            posterior_trees = random.sample(posterior_trees, 1000)
        
        num_leaves = dataset.return_number_leaves(0)
        sampled = [str(Tree(num_leaves=num_leaves, random=True)) for _ in range(len(posterior_trees))]
        res = compare_branch_length_distributions(posterior_trees, sampled)
        self.assertGreater(res['kl_divergence_branch_length'], 0)
        self.assertGreater(res['js_divergence_branch_length'], 0)


if __name__ == "__main__":
    unittest.main()
