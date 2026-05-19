import unittest

import torch

from model.treeTokenizer import TreeFeatureTokenizer


class TreeTokenizerPolytomyLapPETest(unittest.TestCase):
    def test_full_adjacency_lappe_preserves_polytomy_children(self):
        tokenizer = TreeFeatureTokenizer(
            num_node_types=3,
            num_edge_types=3,
            hidden_dim=32,
            lap_dim=4,
        )
        tokenizer.eval()

        # One internal node has four children; the old path truncated this to two.
        structural = tokenizer._newick_to_structural(
            "((((6:0,7:0):1,2:0,3:0,4:0):1,(0:0,1:0):1));"
        )
        child_ptr = torch.from_numpy(structural[0]).long()
        child_ids = torch.from_numpy(structural[1]).long()
        parent_arr = torch.from_numpy(structural[2]).long()
        child_arr = torch.from_numpy(structural[3]).long()
        root_idx = structural[4]
        branch_lengths = torch.from_numpy(structural[5]).float()
        edge_types = torch.from_numpy(structural[6]).long()

        (
            _node_data,
            edge_index,
            _edge_data,
            _branch_lengths,
            node_num,
            _edge_num,
            lap_full,
            _leaf_mask,
            _leaf_idx,
            _sin_embed_node,
            _sin_embed_edge,
        ) = tokenizer.tree_to_graph_from_children(
            child_ptr,
            child_ids,
            parent_arr,
            child_arr,
            root_idx,
            branch_lengths,
            edge_types,
        )

        lap_direct = tokenizer.lap_pe_from_edge_index(
            edge_index=edge_index,
            num_nodes=node_num,
            k=tokenizer.lap_dim,
            device=edge_index.device,
        )
        self.assertTrue(torch.allclose(lap_full, lap_direct))

        truncated_children = torch.full((node_num, 2), -1, dtype=torch.long)
        for p in range(node_num):
            s = child_ptr[p].item()
            t = child_ptr[p + 1].item()
            c = child_ids[s:t]
            if c.numel() > 0:
                truncated_children[p, 0] = c[0]
            if c.numel() > 1:
                truncated_children[p, 1] = c[1]

        lap_truncated = tokenizer.lap_pe_torch(
            truncated_children,
            k=tokenizer.lap_dim,
            device=truncated_children.device,
        )

        # Compare pairwise distances to avoid eigenvector sign ambiguity.
        dist_full = torch.cdist(lap_full, lap_full)
        dist_truncated = torch.cdist(lap_truncated, lap_truncated)
        self.assertFalse(
            torch.allclose(dist_full, dist_truncated, atol=1e-4, rtol=1e-4)
        )


if __name__ == "__main__":
    unittest.main()
