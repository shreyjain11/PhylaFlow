import random

import numpy as np
import torch
import torch.nn as nn
from ete3 import Tree as EteTree

from run.training_helpers import *


class BranchRelaxHead(nn.Module):
    def __init__(
        self,
        edge_dim: int,
        num_cases: int,
        *,
        case_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.case_embedding = nn.Embedding(max(1, int(num_cases)), int(case_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(int(edge_dim) + int(case_dim) + 3),
            nn.Linear(int(edge_dim) + int(case_dim) + 3, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, edge_features, numeric_features, case_indices):
        case_indices = torch.clamp(
            case_indices.to(device=edge_features.device, dtype=torch.long),
            min=0,
            max=self.case_embedding.num_embeddings - 1,
        )
        case_features = self.case_embedding(case_indices)
        x = torch.cat([edge_features, numeric_features, case_features], dim=-1)
        return self.net(x).squeeze(-1)


class BranchRelaxerMixin:
    def _branch_relax_training_loss(self):
        if (
            self.branch_relax_head is None
            or self.branch_relax_head_weight <= 0.0
            or not self.branch_relax_samples
        ):
            return None, {}
        batch_size = min(int(self.branch_relax_batch_size), len(self.branch_relax_samples))
        samples = random.sample(self.branch_relax_samples, k=batch_size)
        tokenized = _move_tokenized_batch_to_device(
            self.model.tokenizer([sample["newick_tree"] for sample in samples]),
            self.device,
        )
        case_indices = torch.tensor(
            [int(sample["case_index"]) for sample in samples],
            dtype=torch.long,
            device=self.device,
        )
        (
            _velocity,
            edge_splits,
            _edge_mask,
            _first_hit_logits,
            _boundary_vanish_logits,
            edge_features,
        ) = self.forward(
            tokenized,
            torch.tensor([4.0], dtype=torch.float32, device=self.device),
            None,
            first_hit_case_indices=case_indices,
        )
        if edge_features is None:
            return None, {}
        preds = []
        labels = []
        for batch_idx, sample in enumerate(samples):
            entries, _lengths, _n_leaves, _mapping = _branch_relax_entries_for_tree(
                self,
                sample["newick_tree"],
                edge_splits[batch_idx],
                labels=sample["labels"],
            )
            if not entries:
                continue
            feature_block = torch.stack(
                [edge_features[batch_idx, entry["edge_index"]] for entry in entries],
                dim=0,
            )
            if self.branch_relax_detach_trunk:
                feature_block = feature_block.detach()
            numeric = torch.tensor(
                [entry["numeric"] for entry in entries],
                dtype=torch.float32,
                device=self.device,
            )
            case_block = torch.full(
                (len(entries),),
                int(sample["case_index"]),
                dtype=torch.long,
                device=self.device,
            )
            preds.append(self.branch_relax_head(feature_block, numeric, case_block))
            labels.append(
                torch.tensor(
                    [float(entry["label"]) for entry in entries],
                    dtype=torch.float32,
                    device=self.device,
                )
            )
        if not preds:
            return None, {}
        pred = torch.cat(preds)
        target = torch.cat(labels)
        diff = pred - target
        loss = diff.pow(2).mean()
        logs = {
            "train/branch_relax_loss_unscaled": loss.detach(),
            "train/branch_relax_mae": diff.abs().mean().detach(),
            "train/branch_relax_sign_acc": (
                ((pred > 0.0) == (target > 0.0)).float().mean().detach()
            ),
        }
        return loss, logs
