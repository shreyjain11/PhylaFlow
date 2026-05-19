import math
from collections import OrderedDict
from math import log, sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.tree_transformer_layers import (
    DropPath,
    MultiheadAttention,
    MultiheadPerformerAttention,
)
from model.treeTokenizer import TreeFeatureTokenizer
from utils.bhv_utils import (
    get_batch_explicit_structural_group_indices,
    get_batch_structural_polytomy_indices,
)


def _load_frozen_start_case_embedding_table(path, *, name):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    table = payload.get("embeddings") if isinstance(payload, dict) else payload
    if table is None:
        raise ValueError(f"{name} artifact at {path} does not contain embeddings")
    table = torch.as_tensor(table, dtype=torch.float32)
    if table.ndim != 2:
        raise ValueError(
            f"{name} embeddings must have shape [num_cases, dim], got {tuple(table.shape)}"
        )
    if int(table.shape[0]) <= 0 or int(table.shape[1]) <= 0:
        raise ValueError(f"{name} embeddings cannot be empty")
    return table.contiguous()


# TokenGT parameter initialization
def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02 / sqrt(n_layers))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if hasattr(module, "padding_idx") and module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])


class PairwiseMergeHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden: int = 256,
        dropout: float = 0.1,
        context_dim: int = 0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.context_dim = int(context_dim)

        in_dim = 4 * d_model + self.context_dim  # [hi, hj, |hi-hj|, hi*hj, ctx]
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),  # logit
        )

    def forward(self, H: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """
        H: [G, D]
        returns logits: [G, G] with -inf on diagonal (no self-merge)
        """
        H = self.norm(H)
        G, D = H.shape
        if self.context_dim:
            if context is None:
                raise ValueError("context is required for context-aware pairwise head")
            context = context.to(device=H.device, dtype=H.dtype)
            if context.ndim != 1 or context.shape[0] != self.context_dim:
                raise ValueError(
                    f"context must have shape [{self.context_dim}] for pairwise head"
                )
        hi = H.unsqueeze(1).expand(G, G, D)  # [G, G, D]
        hj = H.unsqueeze(0).expand(G, G, D)  # [G, G, D]
        feat_parts = [hi, hj, (hi - hj).abs(), hi * hj]
        if self.context_dim:
            feat_parts.append(context.view(1, 1, -1).expand(G, G, -1))
        feats = torch.cat(feat_parts, dim=-1)
        logits = self.mlp(feats).squeeze(-1)  # [G, G]

        # disallow i==j
        logits = logits.masked_fill(
            torch.eye(G, device=H.device, dtype=torch.bool), float("-inf")
        )
        return logits


class StructuredSubsetMergeHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden: int = 256,
        dropout: float = 0.1,
        max_subset_size: int = 64,
        context_dim: int = 0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.max_subset_size = int(max_subset_size)
        self.context_dim = int(context_dim)
        pair_in_dim = 4 * d_model + self.context_dim
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pair_logit = nn.Linear(hidden, 1)
        self.pair_context_proj = nn.Sequential(
            nn.Linear(hidden + d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        member_in_dim = 4 * d_model + self.context_dim
        self.member_mlp = nn.Sequential(
            nn.Linear(member_in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        global_in_dim = d_model + self.context_dim
        self.subset_size_head = nn.Sequential(
            nn.Linear(global_in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.max_subset_size + 1),
        )
        self.stop_after_merge_head = nn.Sequential(
            nn.Linear(global_in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        H: torch.Tensor,
        context: torch.Tensor | None = None,
        top_member_pairs: int | None = None,
    ) -> dict:
        """
        H: [G, D]
        Returns:
          starter_pair_logits: [P]
          starter_pair_indices: list[(i, j)] for unordered pairs
          member_logits: [P, G] logits for including each component in the subset
          logits: [G, G] symmetric starter-pair score matrix for compatibility
        """
        H = self.norm(H)
        G, D = H.shape
        if self.context_dim:
            if context is None:
                raise ValueError("context is required for context-aware structured head")
            context = context.to(device=H.device, dtype=H.dtype)
            if context.ndim != 1 or context.shape[0] != self.context_dim:
                raise ValueError(
                    f"context must have shape [{self.context_dim}] for structured head"
                )
        if G <= 1:
            empty_logits = H.new_empty((0,))
            return {
                "starter_pair_logits": empty_logits,
                "starter_pair_indices": [],
                "member_logits": H.new_empty((0, G)),
                "logits": H.new_full((G, G), float("-inf")),
                "subset_size_logits": H.new_full(
                    (self.max_subset_size + 1,),
                    float("-inf"),
                ),
                "stop_after_merge_logit": H.new_zeros(()),
            }

        pair_indices = torch.triu_indices(G, G, offset=1, device=H.device)
        left_idx, right_idx = pair_indices[0], pair_indices[1]
        hi = H[left_idx]
        hj = H[right_idx]
        pair_feat_parts = [hi, hj, (hi - hj).abs(), hi * hj]
        if self.context_dim:
            pair_feat_parts.append(context.unsqueeze(0).expand(hi.size(0), -1))
        pair_feats = torch.cat(pair_feat_parts, dim=-1)
        pair_hidden = self.pair_mlp(pair_feats)
        starter_pair_logits = self.pair_logit(pair_hidden).squeeze(-1)

        pair_summary = 0.5 * (hi + hj)
        selected_pair_ids = None
        if top_member_pairs is not None:
            top_member_pairs = int(top_member_pairs)
            if top_member_pairs > 0 and starter_pair_logits.numel() > top_member_pairs:
                if top_member_pairs == 1:
                    selected_pair_ids = torch.argmax(
                        starter_pair_logits.detach(),
                    ).reshape(1)
                else:
                    selected_pair_ids = torch.topk(
                        starter_pair_logits.detach(),
                        k=top_member_pairs,
                    ).indices

        if selected_pair_ids is None:
            member_pair_ids = None
            pair_context_source = pair_hidden
            pair_summary_source = pair_summary
        else:
            member_pair_ids = selected_pair_ids
            pair_context_source = pair_hidden.index_select(0, member_pair_ids)
            pair_summary_source = pair_summary.index_select(0, member_pair_ids)

        pair_context = self.pair_context_proj(
            torch.cat([pair_context_source, pair_summary_source], dim=-1)
        )

        node_expand = H.unsqueeze(0).expand(pair_context.size(0), G, D)
        pair_expand = pair_context.unsqueeze(1).expand(pair_context.size(0), G, D)
        member_feat_parts = [
            node_expand,
            pair_expand,
            (node_expand - pair_expand).abs(),
            node_expand * pair_expand,
        ]
        if self.context_dim:
            member_feat_parts.append(
                context.view(1, 1, -1).expand(pair_context.size(0), G, -1)
            )
        member_feats = torch.cat(member_feat_parts, dim=-1)
        member_logits_computed = self.member_mlp(member_feats).squeeze(-1)
        if selected_pair_ids is None:
            member_logits = member_logits_computed
        else:
            member_logits = H.new_full(
                (starter_pair_logits.size(0), G),
                float("-inf"),
            )
            member_logits[member_pair_ids] = member_logits_computed

        pair_matrix = H.new_full((G, G), float("-inf"))
        pair_matrix[left_idx, right_idx] = starter_pair_logits
        pair_matrix[right_idx, left_idx] = starter_pair_logits

        pooled = H.mean(dim=0)
        if self.context_dim:
            pooled = torch.cat([pooled, context], dim=-1)

        pair_indices_cpu = torch.triu_indices(G, G, offset=1, device="cpu")
        starter_pair_indices = list(
            zip(pair_indices_cpu[0].tolist(), pair_indices_cpu[1].tolist())
        )

        return {
            "starter_pair_logits": starter_pair_logits,
            "starter_pair_indices": starter_pair_indices,
            "member_logits": member_logits,
            "logits": pair_matrix,
            "subset_size_logits": self.subset_size_head(pooled),
            "stop_after_merge_logit": self.stop_after_merge_head(pooled).squeeze(-1),
        }


class AutoregressiveGroupRefinementBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        ffn_mult: int = 4,
        n_layers: int = 12,
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            n_layers=n_layers,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.apply(lambda m: init_params(m, n_layers))

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        if H.size(0) <= 1:
            return H

        attn_input = self.attn_norm(H).unsqueeze(0)
        attn_output, _ = self.attn(attn_input)
        H = H + self.attn_dropout(attn_output.squeeze(0))
        H = H + self.ffn(self.ffn_norm(H))
        return H


class TreeGraphEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        n_heads,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        drop_path=0.0,
        use_performer=False,
        performer_nb_features=None,
        performer_generalized_attention=False,
        layernorm_style="prenorm",
        n_layers=12,
    ):
        super().__init__()
        self.layernorm_style = layernorm_style
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if use_performer and performer_nb_features is not None:
            self.self_attn = MultiheadPerformerAttention(
                embed_dim,
                n_heads,
                performer_nb_features=performer_nb_features,
                performer_generalized_attention=performer_generalized_attention,
                dropout=dropout,
                n_layers=n_layers,
            )
        else:
            self.self_attn = MultiheadAttention(
                embed_dim,
                n_heads,
                dropout=attention_dropout,
                n_layers=n_layers,
            )
        self.dropout_module = nn.Dropout(dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(activation_dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.apply(lambda m: init_params(m, n_layers))

    def forward(self, x, padding_mask=None):  # padding_mask is key_padding_mask
        attn = None
        if self.layernorm_style == "prenorm":
            residual = x
            x_norm = self.self_attn_layer_norm(x)
            x_attn, attn = self.self_attn(x_norm, key_padding_mask=padding_mask)
            x_attn = self.dropout_module(x_attn)
            x = self.drop_path1(x_attn)
            x = residual + x

            residual = x
            x_norm = self.final_layer_norm(x)
            x_ffn = self.feedforward(x_norm)
            x = self.drop_path2(x_ffn)
            x = residual + x
        elif self.layernorm_style == "postnorm":
            residual = x
            x_attn, attn = self.self_attn(x, key_padding_mask=padding_mask)
            x_attn = self.dropout_module(x_attn)
            x = self.drop_path1(x_attn)
            x = residual + x
            x = self.self_attn_layer_norm(x)

            residual = x
            x_ffn = self.feedforward(x)
            x = self.drop_path2(x_ffn)
            x = residual + x
            x = self.final_layer_norm(x)
        else:
            raise NotImplementedError
        return x, attn


class TreeDenoiserTokenGT(nn.Module):
    def __init__(
        self,
        num_node_types,
        num_edge_types,
        embed_dim=768,
        n_layers=12,
        n_heads=32,
        output_dim=1,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        drop_path_rate=0.1,
        use_performer=True,
        performer_nb_features=64,
        performer_generalized_attention=True,
        layernorm_style="prenorm",
        tokenizer_lap_dim=16,  # TreeFeatureTokenizer
        tokenizer_lap_dropout=0.2,  # TreeFeatureTokenizer
        tokenizer_n_layers=6,  # TreeFeatureTokenizer
        tokenizer_branch_length_mode="linear",
        tokenizer_branch_length_num_buckets=64,
        tokenizer_branch_length_log_min=-8.0,
        tokenizer_branch_length_log_max=1.0,
        phyla_dim=256,
        phyla_use_leaf_tokens=True,
        phyla_use_split_tokens=True,
        phyla_leaf_scale=1.0,
        phyla_split_scale=1.0,
        phyla_use_global_context=False,
        phyla_global_context_scale=1.0,
        phyla_use_clade_context=False,
        phyla_clade_context_scale=1.0,
        autoregressive_head_mode="pairwise_threshold",
        autoregressive_group_refinement_layers=0,
        autoregressive_max_subset_size=64,
        autoregressive_use_case_conditioning=False,
        autoregressive_num_cases=None,
        autoregressive_case_dim=16,
        autoregressive_use_start_topology_conditioning=False,
        autoregressive_start_topology_hidden_dim=None,
        autoregressive_start_topology_conditioning_mode="additive",
        autoregressive_start_topology_code_dim=64,
        autoregressive_frozen_start_case_embedding_path=None,
        autoregressive_frozen_start_case_adapter_mode="linear",
        autoregressive_frozen_start_case_adapter_hidden_dim=None,
        first_hit_head_use_phase_input=False,
        first_hit_head_phase_hidden_dim=None,
        first_hit_head_mode="base",
        first_hit_head_hidden_dim=None,
        first_hit_head_enable_refinement=False,
        first_hit_head_refinement_layers=1,
        first_hit_head_router_hidden_dim=None,
        first_hit_head_num_cases=None,
        first_hit_head_case_dim=16,
        first_hit_frozen_start_case_embedding_path=None,
        first_hit_frozen_start_case_adapter_mode="linear",
        first_hit_frozen_start_case_adapter_hidden_dim=None,
        first_hit_start_tree_graph_detach=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.tokenizer = TreeFeatureTokenizer(
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            hidden_dim=embed_dim,
            n_layers=tokenizer_n_layers,
            lap_dim=tokenizer_lap_dim,
            lap_dropout=tokenizer_lap_dropout,
            branch_length_mode=tokenizer_branch_length_mode,
            branch_length_num_buckets=tokenizer_branch_length_num_buckets,
            branch_length_log_min=tokenizer_branch_length_log_min,
            branch_length_log_max=tokenizer_branch_length_log_max,
            # concat_features=True,  # Use concatenation of features
        )
        # [graph] token and [null] token
        self.graph_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.null_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.graph_token, mean=0.0, std=0.02)
        nn.init.normal_(self.null_token, mean=0.0, std=0.02)

        # self.embed_proj = nn.Linear(embed_dim, embed_dim)

        # Phyla projection
        self.phyla_proj = nn.Linear(phyla_dim, embed_dim)
        self.phyla_split_proj = nn.Sequential(
            nn.Linear(2 * phyla_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.phyla_use_leaf_tokens = bool(phyla_use_leaf_tokens)
        self.phyla_use_split_tokens = bool(phyla_use_split_tokens)
        self.phyla_use_global_context = bool(phyla_use_global_context)
        self.phyla_use_clade_context = bool(phyla_use_clade_context)
        self.phyla_leaf_scale = float(phyla_leaf_scale)
        self.phyla_split_scale = float(phyla_split_scale)
        self.phyla_global_context_scale = float(phyla_global_context_scale)
        self.phyla_clade_context_scale = float(phyla_clade_context_scale)
        self.phyla_global_proj = None
        if self.phyla_use_global_context:
            self.phyla_global_proj = nn.Sequential(
                nn.LayerNorm(phyla_dim),
                nn.Linear(phyla_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
            )
        self.phyla_clade_proj = None
        if self.phyla_use_clade_context:
            self.phyla_clade_proj = nn.Sequential(
                nn.LayerNorm(2 * phyla_dim),
                nn.Linear(2 * phyla_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
            )

        # Time projection
        self.time_embed_dim = embed_dim * 4
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, self.time_embed_dim),
            nn.GELU(),
            nn.Linear(self.time_embed_dim, embed_dim),
        )

        # Transformer encoder
        dprates = [
            drop_path_rate * i / (n_layers - 1) if n_layers > 1 else 0.0
            for i in range(n_layers)
        ]
        self.layers = nn.ModuleList(
            [
                TreeGraphEncoderLayer(
                    embed_dim=embed_dim,
                    ffn_dim=embed_dim * 4,
                    n_heads=n_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    drop_path=dprates[i],
                    use_performer=use_performer,
                    performer_nb_features=performer_nb_features,
                    performer_generalized_attention=performer_generalized_attention,
                    layernorm_style=layernorm_style,
                    n_layers=n_layers,
                )
                for i in range(n_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.output_layer = nn.Linear(embed_dim, output_dim)

        self.edge_output_layer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, output_dim), # Final layer is LINEAR
        )
        self.first_hit_head_use_phase_input = bool(first_hit_head_use_phase_input)
        self.first_hit_head_phase_hidden_dim = int(
            embed_dim
            if first_hit_head_phase_hidden_dim is None
            else first_hit_head_phase_hidden_dim
        )
        self.first_hit_head_mode = str(first_hit_head_mode)
        self.first_hit_head_hidden_dim = int(
            embed_dim
            if first_hit_head_hidden_dim is None
            else first_hit_head_hidden_dim
        )
        self.first_hit_head_enable_refinement = bool(
            first_hit_head_enable_refinement
        ) or self.first_hit_head_mode == "edge_refined_mlp"
        self.first_hit_head_refinement_layers = max(1, int(first_hit_head_refinement_layers))
        self.first_hit_head_router_hidden_dim = int(
            embed_dim
            if first_hit_head_router_hidden_dim is None
            else first_hit_head_router_hidden_dim
        )
        self.first_hit_head_num_cases = (
            None
            if first_hit_head_num_cases is None
            else int(first_hit_head_num_cases)
        )
        self.first_hit_head_case_dim = int(first_hit_head_case_dim)
        self.first_hit_frozen_start_case_embedding_path = (
            first_hit_frozen_start_case_embedding_path
        )
        self.first_hit_frozen_start_case_adapter_mode = str(
            first_hit_frozen_start_case_adapter_mode
        )
        if self.first_hit_frozen_start_case_adapter_mode not in {"linear", "mlp2"}:
            raise ValueError(
                "first_hit_frozen_start_case_adapter_mode must be one of "
                "['linear', 'mlp2']"
            )
        self.first_hit_frozen_start_case_adapter_hidden_dim = int(
            self.first_hit_head_hidden_dim
            if first_hit_frozen_start_case_adapter_hidden_dim is None
            else first_hit_frozen_start_case_adapter_hidden_dim
        )
        self.first_hit_start_tree_graph_detach = bool(
            first_hit_start_tree_graph_detach
        )
        self.first_hit_phase_proj = None
        first_hit_head_input_dim = embed_dim
        if self.first_hit_head_use_phase_input:
            self.first_hit_phase_proj = nn.Sequential(
                nn.Linear(1, self.first_hit_head_phase_hidden_dim),
                nn.GELU(),
                nn.Linear(self.first_hit_head_phase_hidden_dim, embed_dim),
            )
            first_hit_head_input_dim = 2 * embed_dim
        self.first_hit_phyla_global_proj = None
        if self.phyla_use_global_context:
            self.first_hit_phyla_global_proj = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, first_hit_head_input_dim),
            )
        self.first_hit_edge_head = nn.Sequential(
            nn.LayerNorm(first_hit_head_input_dim),
            nn.Linear(first_hit_head_input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        self.first_hit_edge_head_shared = None
        self.first_hit_edge_head_out = None
        self.first_hit_edge_head_router = None
        self.first_hit_tree_context_proj = None
        self.first_hit_edge_head_conditioned = None
        self.first_hit_case_embedding = None
        self.first_hit_topology_adapter = None
        self.first_hit_start_topology_adapter = None
        self.first_hit_edge_head_adapted = None
        self.first_hit_topology_cross_query = None
        self.first_hit_topology_cross_key = None
        self.first_hit_topology_cross_value = None
        self.first_hit_frozen_start_case_proj = None
        self.first_hit_edge_head_cross_attn = None
        self.first_hit_edge_head_raw_topology = None
        self.first_hit_edge_head_start_topology_raw = None
        self.first_hit_edge_head_start_tree_graph = None
        self.first_hit_edge_refinement = None
        self.first_hit_autoregressive_edge_encoder = None
        self.first_hit_autoregressive_query = None
        self.first_hit_autoregressive_key = None
        self.first_hit_autoregressive_update = None
        self.first_hit_autoregressive_stop = None
        self.first_hit_autoregressive_start = None
        if self.first_hit_head_enable_refinement:
            self.first_hit_edge_refinement = nn.ModuleList(
                [
                    AutoregressiveGroupRefinementBlock(
                        d_model=embed_dim,
                        num_heads=n_heads,
                        dropout=dropout,
                        ffn_mult=4,
                        n_layers=n_layers,
                    )
                    for _ in range(self.first_hit_head_refinement_layers)
                ]
            )
        if self.first_hit_head_mode in {
            "shared_mlp",
            "routed_adapter",
            "tree_conditioned_mlp",
        }:
            self.first_hit_edge_head_shared = nn.Sequential(
                nn.LayerNorm(first_hit_head_input_dim),
                nn.Linear(first_hit_head_input_dim, self.first_hit_head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.first_hit_head_hidden_dim, self.first_hit_head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.first_hit_edge_head_out = nn.Linear(self.first_hit_head_hidden_dim, 1)
        if self.first_hit_head_mode == "routed_adapter":
            router_input_dim = first_hit_head_input_dim + embed_dim
            self.first_hit_edge_head_router = nn.Sequential(
                nn.LayerNorm(router_input_dim),
                nn.Linear(router_input_dim, self.first_hit_head_router_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(
                    self.first_hit_head_router_hidden_dim,
                    2 * self.first_hit_head_hidden_dim,
                ),
            )
        if self.first_hit_head_mode == "tree_conditioned_mlp":
            tree_context_input_dim = first_hit_head_input_dim + embed_dim
            self.first_hit_tree_context_proj = nn.Sequential(
                nn.LayerNorm(tree_context_input_dim),
                nn.Linear(tree_context_input_dim, self.first_hit_head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            conditioned_input_dim = first_hit_head_input_dim + self.first_hit_head_hidden_dim
            self.first_hit_edge_head_conditioned = nn.Sequential(
                nn.LayerNorm(conditioned_input_dim),
                nn.Linear(conditioned_input_dim, self.first_hit_head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.first_hit_head_hidden_dim, self.first_hit_head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.first_hit_head_hidden_dim, 1),
            )
        if self.first_hit_head_mode in {
            "case_adapted_mlp",
            "frozen_start_case_mlp",
            "topology_adapter_mlp",
            "topology_attention_adapter_mlp",
            "start_topology_adapter_mlp",
        }:
            adapted_input_dim = first_hit_head_input_dim + self.first_hit_head_case_dim
            self.first_hit_edge_head_adapted = nn.Sequential(
                nn.LayerNorm(adapted_input_dim),
                nn.Linear(adapted_input_dim, self.first_hit_head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.first_hit_head_hidden_dim, 1),
            )
        if self.first_hit_head_mode == "topology_raw_pool_concat_mlp":
            raw_topology_input_dim = first_hit_head_input_dim + (2 * embed_dim)
            self.first_hit_edge_head_raw_topology = nn.Sequential(
                nn.LayerNorm(raw_topology_input_dim),
                nn.Linear(raw_topology_input_dim, self.first_hit_head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.first_hit_head_hidden_dim, self.first_hit_head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.first_hit_head_hidden_dim, 1),
            )
        if self.first_hit_head_mode == "start_topology_raw_pool_concat_mlp":
            start_topology_raw_input_dim = first_hit_head_input_dim + (3 * embed_dim)
            self.first_hit_edge_head_start_topology_raw = nn.Sequential(
                nn.LayerNorm(start_topology_raw_input_dim),
                nn.Linear(
                    start_topology_raw_input_dim,
                    self.first_hit_head_hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(
                    self.first_hit_head_hidden_dim,
                    self.first_hit_head_hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.first_hit_head_hidden_dim, 1),
            )
        if self.first_hit_head_mode == "start_tree_graph_token_mlp":
            start_tree_graph_input_dim = first_hit_head_input_dim + embed_dim
            self.first_hit_edge_head_start_tree_graph = nn.Sequential(
                nn.LayerNorm(start_tree_graph_input_dim),
                nn.Linear(start_tree_graph_input_dim, self.first_hit_head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.first_hit_head_hidden_dim, self.first_hit_head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.first_hit_head_hidden_dim, 1),
            )
        if self.first_hit_head_mode in {
            "topology_cross_attn_mlp",
            "start_topology_cross_attn_mlp",
        }:
            self.first_hit_topology_cross_query = nn.Sequential(
                nn.LayerNorm(first_hit_head_input_dim + embed_dim),
                nn.Linear(first_hit_head_input_dim + embed_dim, self.first_hit_head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.first_hit_head_hidden_dim, self.first_hit_head_hidden_dim),
            )
            self.first_hit_topology_cross_key = nn.Linear(
                embed_dim,
                self.first_hit_head_hidden_dim,
                bias=False,
            )
            self.first_hit_topology_cross_value = nn.Linear(
                embed_dim,
                self.first_hit_head_hidden_dim,
                bias=False,
            )
            cross_attn_input_dim = (
                first_hit_head_input_dim + embed_dim + self.first_hit_head_hidden_dim
            )
            self.first_hit_edge_head_cross_attn = nn.Sequential(
                nn.LayerNorm(cross_attn_input_dim),
                nn.Linear(cross_attn_input_dim, self.first_hit_head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.first_hit_head_hidden_dim, self.first_hit_head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.first_hit_head_hidden_dim, 1),
            )
        if self.first_hit_head_mode == "case_adapted_mlp":
            if self.first_hit_head_num_cases is None or self.first_hit_head_num_cases <= 0:
                raise ValueError(
                    "first_hit_head_num_cases must be set for case_adapted_mlp"
                )
            self.first_hit_case_embedding = nn.Embedding(
                self.first_hit_head_num_cases,
                self.first_hit_head_case_dim,
            )
        if self.first_hit_head_mode == "frozen_start_case_mlp":
            if not self.first_hit_frozen_start_case_embedding_path:
                raise ValueError(
                    "first_hit_frozen_start_case_embedding_path must be set for "
                    "frozen_start_case_mlp"
                )
            frozen_table = _load_frozen_start_case_embedding_table(
                self.first_hit_frozen_start_case_embedding_path,
                name="first_hit_frozen_start_case",
            )
            if (
                self.first_hit_head_num_cases is not None
                and int(frozen_table.shape[0]) != self.first_hit_head_num_cases
            ):
                raise ValueError(
                    "first_hit_frozen_start_case num cases "
                    f"{int(frozen_table.shape[0])} does not match "
                    f"first_hit_head_num_cases={self.first_hit_head_num_cases}"
                )
            self.first_hit_head_num_cases = int(frozen_table.shape[0])
            frozen_dim = int(frozen_table.shape[1])
            if self.first_hit_frozen_start_case_adapter_mode == "mlp2":
                self.first_hit_frozen_start_case_proj = nn.Sequential(
                    nn.LayerNorm(frozen_dim),
                    nn.Linear(
                        frozen_dim,
                        self.first_hit_frozen_start_case_adapter_hidden_dim,
                    ),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(
                        self.first_hit_frozen_start_case_adapter_hidden_dim,
                        self.first_hit_head_case_dim,
                    ),
                )
            elif frozen_dim != self.first_hit_head_case_dim:
                self.first_hit_frozen_start_case_proj = nn.Sequential(
                    nn.LayerNorm(frozen_dim),
                    nn.Linear(frozen_dim, self.first_hit_head_case_dim),
                )
            self.register_buffer(
                "first_hit_frozen_start_case_embedding",
                frozen_table,
                persistent=True,
            )
        if self.first_hit_head_mode == "topology_adapter_mlp":
            topology_context_input_dim = (3 * embed_dim)
            self.first_hit_topology_adapter = nn.Sequential(
                nn.LayerNorm(topology_context_input_dim),
                nn.Linear(topology_context_input_dim, self.first_hit_head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.first_hit_head_hidden_dim, self.first_hit_head_case_dim),
            )
        if self.first_hit_head_mode == "start_topology_adapter_mlp":
            start_topology_context_input_dim = 4 * embed_dim
            self.first_hit_start_topology_adapter = nn.Sequential(
                nn.LayerNorm(start_topology_context_input_dim),
                nn.Linear(start_topology_context_input_dim, self.first_hit_head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.first_hit_head_hidden_dim, self.first_hit_head_case_dim),
            )
        self.first_hit_topology_query = None
        self.first_hit_topology_key = None
        self.first_hit_topology_value = None
        if self.first_hit_head_mode == "topology_attention_adapter_mlp":
            topology_query_input_dim = first_hit_head_input_dim + embed_dim
            self.first_hit_topology_query = nn.Sequential(
                nn.LayerNorm(topology_query_input_dim),
                nn.Linear(topology_query_input_dim, self.first_hit_head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.first_hit_head_hidden_dim, self.first_hit_head_case_dim),
            )
            self.first_hit_topology_key = nn.Linear(
                embed_dim,
                self.first_hit_head_case_dim,
                bias=False,
            )
            self.first_hit_topology_value = nn.Linear(
                embed_dim,
                self.first_hit_head_case_dim,
                bias=False,
            )
            topology_context_input_dim = 4 * self.first_hit_head_case_dim
            self.first_hit_topology_adapter = nn.Sequential(
                nn.LayerNorm(topology_context_input_dim),
                nn.Linear(topology_context_input_dim, self.first_hit_head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.first_hit_head_hidden_dim, self.first_hit_head_case_dim),
            )
        if self.first_hit_head_mode == "autoregressive_set":
            self.first_hit_autoregressive_edge_encoder = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, self.first_hit_head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(
                    self.first_hit_head_hidden_dim,
                    self.first_hit_head_hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.first_hit_autoregressive_query = nn.Linear(
                self.first_hit_head_hidden_dim,
                self.first_hit_head_hidden_dim,
                bias=False,
            )
            self.first_hit_autoregressive_key = nn.Linear(
                self.first_hit_head_hidden_dim,
                self.first_hit_head_hidden_dim,
                bias=False,
            )
            self.first_hit_autoregressive_update = nn.GRUCell(
                self.first_hit_head_hidden_dim,
                self.first_hit_head_hidden_dim,
            )
            self.first_hit_autoregressive_stop = nn.Sequential(
                nn.LayerNorm(self.first_hit_head_hidden_dim),
                nn.Linear(self.first_hit_head_hidden_dim, 1),
            )
            self.first_hit_autoregressive_start = nn.Parameter(
                torch.zeros(self.first_hit_head_hidden_dim)
            )
        self.boundary_vanish_edge_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        #This used to be 0.2
        self.edge_output_scale = 2
        self.max_split_bits = 256
        self.split_mask_proj = nn.Sequential(
            nn.Linear(self.max_split_bits, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # self.split_output_layer = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.GELU(),
        #     nn.Linear(embed_dim, output_dim),
        # )
        self.split_identity_scale = 0.75
        self.split_output_scale = 0.5
        self._split_identity_binary_cache = OrderedDict()
        self._split_identity_binary_cache_max = 4096
        self.dropout = nn.Dropout(dropout)
        self.autoregressive_head_mode = str(autoregressive_head_mode)
        self.autoregressive_group_refinement_layers = int(
            autoregressive_group_refinement_layers
        )
        self.autoregressive_max_subset_size = int(autoregressive_max_subset_size)
        self.autoregressive_use_case_conditioning = bool(
            autoregressive_use_case_conditioning
        )
        self.autoregressive_use_start_topology_conditioning = bool(
            autoregressive_use_start_topology_conditioning
        )
        if (
            self.autoregressive_use_case_conditioning
            and self.autoregressive_use_start_topology_conditioning
        ):
            raise ValueError(
                "Use either autoregressive case conditioning or start-topology "
                "conditioning, not both."
            )
        self.autoregressive_num_cases = (
            None if autoregressive_num_cases is None else int(autoregressive_num_cases)
        )
        self.autoregressive_case_dim = int(autoregressive_case_dim)
        self.autoregressive_start_topology_hidden_dim = int(
            embed_dim
            if autoregressive_start_topology_hidden_dim is None
            else autoregressive_start_topology_hidden_dim
        )
        self.autoregressive_start_topology_conditioning_mode = str(
            autoregressive_start_topology_conditioning_mode
        )
        valid_start_topology_modes = {
            "additive",
            "head_concat",
            "frozen_case_probe",
            "frozen_case_probe_additive",
        }
        if (
            self.autoregressive_start_topology_conditioning_mode
            not in valid_start_topology_modes
        ):
            raise ValueError(
                "autoregressive_start_topology_conditioning_mode must be one of "
                f"{sorted(valid_start_topology_modes)}"
            )
        self.autoregressive_start_topology_code_dim = int(
            autoregressive_start_topology_code_dim
        )
        self.autoregressive_frozen_start_case_embedding_path = (
            autoregressive_frozen_start_case_embedding_path
        )
        self.autoregressive_frozen_start_case_adapter_mode = str(
            autoregressive_frozen_start_case_adapter_mode
        )
        if self.autoregressive_frozen_start_case_adapter_mode not in {"linear", "mlp2"}:
            raise ValueError(
                "autoregressive_frozen_start_case_adapter_mode must be one of "
                "['linear', 'mlp2']"
            )
        self.autoregressive_frozen_start_case_adapter_hidden_dim = int(
            2 * embed_dim
            if autoregressive_frozen_start_case_adapter_hidden_dim is None
            else autoregressive_frozen_start_case_adapter_hidden_dim
        )
        self.autoregressive_case_embedding = None
        self.autoregressive_case_proj = None
        self.autoregressive_start_topology_adapter = None
        self.autoregressive_start_topology_head_proj = None
        self.autoregressive_frozen_start_case_proj = None
        self.autoregressive_phyla_global_proj = None
        if self.phyla_use_global_context:
            self.autoregressive_phyla_global_proj = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
            )
        if self.autoregressive_use_case_conditioning:
            if self.autoregressive_num_cases is None or self.autoregressive_num_cases <= 0:
                raise ValueError(
                    "autoregressive_num_cases must be set when "
                    "autoregressive_use_case_conditioning is enabled"
                )
            self.autoregressive_case_embedding = nn.Embedding(
                self.autoregressive_num_cases,
                self.autoregressive_case_dim,
            )
            self.autoregressive_case_proj = nn.Sequential(
                nn.LayerNorm(self.autoregressive_case_dim),
                nn.Linear(self.autoregressive_case_dim, embed_dim),
            )
        if (
            self.autoregressive_use_start_topology_conditioning
            and self.autoregressive_start_topology_conditioning_mode == "additive"
        ):
            self.autoregressive_start_topology_adapter = nn.Sequential(
                nn.LayerNorm(3 * embed_dim),
                nn.Linear(
                    3 * embed_dim,
                    self.autoregressive_start_topology_hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.autoregressive_start_topology_hidden_dim, embed_dim),
            )
        if (
            self.autoregressive_use_start_topology_conditioning
            and self.autoregressive_start_topology_conditioning_mode == "head_concat"
        ):
            if self.autoregressive_start_topology_code_dim <= 0:
                raise ValueError(
                    "autoregressive_start_topology_code_dim must be positive for "
                    "head_concat conditioning"
                )
            self.autoregressive_start_topology_head_proj = nn.Linear(
                3 * embed_dim,
                self.autoregressive_start_topology_code_dim,
            )
        if (
            self.autoregressive_use_start_topology_conditioning
            and self.autoregressive_start_topology_conditioning_mode
            in {"frozen_case_probe", "frozen_case_probe_additive"}
        ):
            if not self.autoregressive_frozen_start_case_embedding_path:
                raise ValueError(
                    "autoregressive_frozen_start_case_embedding_path must be set for "
                    "frozen case-probe conditioning"
                )
            frozen_table = _load_frozen_start_case_embedding_table(
                self.autoregressive_frozen_start_case_embedding_path,
                name="autoregressive_frozen_start_case",
            )
            if int(frozen_table.shape[1]) != self.autoregressive_start_topology_code_dim:
                raise ValueError(
                    "autoregressive_frozen_start_case embedding dim "
                    f"{int(frozen_table.shape[1])} does not match "
                    "autoregressive_start_topology_code_dim="
                    f"{self.autoregressive_start_topology_code_dim}"
                )
            if (
                self.autoregressive_start_topology_conditioning_mode
                == "frozen_case_probe_additive"
            ):
                if self.autoregressive_frozen_start_case_adapter_mode == "mlp2":
                    self.autoregressive_frozen_start_case_proj = nn.Sequential(
                        nn.LayerNorm(self.autoregressive_start_topology_code_dim),
                        nn.Linear(
                            self.autoregressive_start_topology_code_dim,
                            self.autoregressive_frozen_start_case_adapter_hidden_dim,
                        ),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(
                            self.autoregressive_frozen_start_case_adapter_hidden_dim,
                            embed_dim,
                        ),
                    )
                else:
                    self.autoregressive_frozen_start_case_proj = nn.Sequential(
                        nn.LayerNorm(self.autoregressive_start_topology_code_dim),
                        nn.Linear(self.autoregressive_start_topology_code_dim, embed_dim),
                    )
            self.register_buffer(
                "autoregressive_frozen_start_case_embedding",
                frozen_table,
                persistent=True,
            )
        self.autoregressive_group_refinement = nn.ModuleList(
            [
                AutoregressiveGroupRefinementBlock(
                    d_model=embed_dim,
                    num_heads=n_heads,
                    dropout=dropout,
                    ffn_mult=4,
                    n_layers=n_layers,
                )
                for _ in range(self.autoregressive_group_refinement_layers)
            ]
        )
        if self.autoregressive_head_mode == "pairwise_threshold":
            self.pairwise_head = PairwiseMergeHead(
                d_model=embed_dim,
                hidden=embed_dim,
                dropout=dropout,
                context_dim=(
                    self.autoregressive_start_topology_code_dim
                    if (
                        self.autoregressive_use_start_topology_conditioning
                        and self.autoregressive_start_topology_conditioning_mode
                        in {"head_concat", "frozen_case_probe"}
                    )
                    else 0
                ),
            )
            self.structured_subset_head = None
        elif self.autoregressive_head_mode == "structured_subset":
            self.pairwise_head = None
            self.structured_subset_head = StructuredSubsetMergeHead(
                d_model=embed_dim,
                hidden=embed_dim,
                dropout=dropout,
                max_subset_size=self.autoregressive_max_subset_size,
                context_dim=(
                    self.autoregressive_start_topology_code_dim
                    if (
                        self.autoregressive_use_start_topology_conditioning
                        and self.autoregressive_start_topology_conditioning_mode
                        in {"head_concat", "frozen_case_probe"}
                    )
                    else 0
                ),
            )
        else:
            raise ValueError(
                f"Unknown autoregressive_head_mode={self.autoregressive_head_mode!r}"
            )
        self.group_head = nn.Linear(embed_dim, 1)

        self.apply(lambda m: init_params(m, n_layers))

    def create_sinusoidal_embedding(self, t, dim):
        """
        Creates a stable sinusoidal embedding based on the original Transformer paper.
        """
        # t is a tensor of shape [B]
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.float32)
        else:
            t = t.float()

        device = self.graph_token.device
        t = t.to(device)

        half_dim = dim // 2

        # Denominator term: 10000^(2i/d)
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=device)
            * -(math.log(10000.0) / (half_dim - 1))
        )

        # Argument to sin/cos: t / 10000^(2i/d)
        emb = t.unsqueeze(-1) / emb.unsqueeze(0)

        # Final embedding
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))  # Pad last dimension

        return emb

    def _create_split_binary_masks(self, split_masks, device, dtype=None):
        if not split_masks:
            return torch.zeros(
                0,
                self.max_split_bits,
                device=device,
                dtype=self.graph_token.dtype if dtype is None else dtype,
            )

        dtype = self.graph_token.dtype if dtype is None else dtype
        split_masks_tuple = tuple(int(mask) for mask in split_masks)
        cache_key = (
            split_masks_tuple,
            str(device),
            str(dtype),
        )
        binary_masks = self._split_identity_binary_cache.get(cache_key)
        cache_writable = not torch.is_inference_mode_enabled()
        if binary_masks is None or (
            not torch.is_inference_mode_enabled() and binary_masks.is_inference()
        ):
            byte_width = max(1, (int(self.max_split_bits) + 7) // 8)
            bit_limit_mask = (1 << int(self.max_split_bits)) - 1
            mask_bytes = b"".join(
                (int(mask_int) & bit_limit_mask).to_bytes(
                    byte_width,
                    byteorder="little",
                    signed=False,
                )
                for mask_int in split_masks_tuple
            )
            packed = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(
                len(split_masks_tuple),
                byte_width,
            )
            unpacked = np.unpackbits(packed, axis=1, bitorder="little")[
                :, : int(self.max_split_bits)
            ]
            binary_masks = torch.as_tensor(unpacked, device=device, dtype=dtype)
            if cache_writable:
                self._split_identity_binary_cache[cache_key] = binary_masks
                if (
                    len(self._split_identity_binary_cache)
                    > self._split_identity_binary_cache_max
                ):
                    self._split_identity_binary_cache.popitem(last=False)
        else:
            self._split_identity_binary_cache.move_to_end(cache_key)

        return binary_masks

    def create_split_identity_embedding(self, split_masks, device):
        binary_masks = self._create_split_binary_masks(
            split_masks,
            device,
            dtype=self.graph_token.dtype,
        )
        if binary_masks.numel() == 0:
            return torch.zeros(0, self.embed_dim, device=device)
        return self.split_mask_proj(binary_masks)

    def _normalize_phyla_embeddings(self, phyla_embeddings, batch_size):
        if phyla_embeddings is None:
            return None
        if isinstance(phyla_embeddings, list):
            max_len = max(emb.shape[0] for emb in phyla_embeddings)
            padded_embeddings = []
            for emb in phyla_embeddings:
                if emb.shape[0] < max_len:
                    padding = torch.zeros(
                        max_len - emb.shape[0],
                        emb.shape[1],
                        device=emb.device,
                        dtype=emb.dtype,
                    )
                    emb = torch.cat([emb, padding], dim=0)
                padded_embeddings.append(emb)
            phyla_embeddings = torch.stack(padded_embeddings, dim=0)
        elif phyla_embeddings.dim() == 2:
            phyla_embeddings = phyla_embeddings.unsqueeze(0)

        if phyla_embeddings.size(0) == 1 and batch_size > 1:
            phyla_embeddings = phyla_embeddings.expand(batch_size, -1, -1)
        if phyla_embeddings.size(0) != batch_size:
            raise ValueError(
                f"phyla_embeddings batch mismatch: got {phyla_embeddings.size(0)} expected {batch_size}"
            )
        return phyla_embeddings

    def _compute_leaf_phyla_token_additions(
        self,
        phyla_proj_full,
        leaf_idx_list,
        num_tokens,
        device,
        dtype,
    ):
        additions = torch.zeros(
            phyla_proj_full.size(0),
            num_tokens,
            self.embed_dim,
            device=device,
            dtype=dtype,
        )
        for b, leaf_indices in enumerate(leaf_idx_list):
            if leaf_indices.numel() == 0:
                continue
            leaf_count = leaf_indices.numel()
            if phyla_proj_full.size(1) < leaf_count:
                raise ValueError(
                    f"Need {leaf_count} phyla embeddings, got {phyla_proj_full.size(1)}"
                )
            additions[b, leaf_indices] = phyla_proj_full[b, :leaf_count].to(dtype)
        return additions

    def _compute_split_phyla_token_additions(
        self,
        phyla_embeddings,
        leaf_idx_list,
        edge_mask,
        edge_split_masks,
        num_tokens,
        device,
        dtype,
    ):
        additions = torch.zeros(
            phyla_embeddings.size(0),
            num_tokens,
            self.embed_dim,
            device=device,
            dtype=dtype,
        )
        zero_raw = torch.zeros(
            phyla_embeddings.size(-1), device=device, dtype=phyla_embeddings.dtype
        )

        for b, leaf_indices in enumerate(leaf_idx_list):
            if leaf_indices.numel() == 0:
                continue

            leaf_count = leaf_indices.numel()
            if phyla_embeddings.size(1) < leaf_count:
                raise ValueError(
                    f"Need {leaf_count} phyla embeddings, got {phyla_embeddings.size(1)}"
                )

            raw_leaf_embeddings = phyla_embeddings[b, :leaf_count]
            leaf_bits = [int(idx.item()) for idx in leaf_indices]
            edge_positions = torch.nonzero(edge_mask[b], as_tuple=True)[0]
            split_masks_b = edge_split_masks[b]
            edge_count = min(edge_positions.numel(), len(split_masks_b))

            for edge_idx in range(edge_count):
                split_mask = int(split_masks_b[edge_idx])
                if split_mask == 0:
                    continue
                select = torch.tensor(
                    [bool((split_mask >> bit) & 1) for bit in leaf_bits],
                    device=device,
                    dtype=torch.bool,
                )
                inside = (
                    raw_leaf_embeddings[select].mean(dim=0)
                    if select.any()
                    else zero_raw
                )
                outside = (
                    raw_leaf_embeddings[~select].mean(dim=0)
                    if (~select).any()
                    else zero_raw
                )
                split_feature = torch.cat([inside, outside], dim=0)
                additions[b, edge_positions[edge_idx]] = self.phyla_split_proj(
                    split_feature
                ).to(dtype)
        return additions

    def _compute_clade_phyla_token_context(
        self,
        phyla_embeddings,
        leaf_idx_list,
        edge_mask,
        edge_split_masks,
        num_tokens,
        device,
        dtype,
    ):
        if not self.phyla_use_clade_context or self.phyla_clade_proj is None:
            return None
        context = torch.zeros(
            phyla_embeddings.size(0),
            num_tokens,
            self.embed_dim,
            device=device,
            dtype=dtype,
        )
        zero_raw = torch.zeros(
            phyla_embeddings.size(-1), device=device, dtype=phyla_embeddings.dtype
        )

        for b, leaf_indices in enumerate(leaf_idx_list):
            leaf_count = int(leaf_indices.numel())
            if leaf_count <= 0:
                continue
            if phyla_embeddings.size(1) < leaf_count:
                raise ValueError(
                    f"Need {leaf_count} phyla embeddings, got {phyla_embeddings.size(1)}"
                )

            edge_positions = torch.nonzero(edge_mask[b], as_tuple=True)[0]
            split_masks_b = edge_split_masks[b]
            edge_count = min(int(edge_positions.numel()), len(split_masks_b))
            if edge_count <= 0:
                continue

            raw_leaf_embeddings = phyla_embeddings[b, :leaf_count].to(device=device)
            leaf_bits = [int(idx) for idx in leaf_indices.detach().cpu().tolist()]
            split_mask_slice = split_masks_b[:edge_count]
            if torch.is_tensor(split_mask_slice):
                split_mask_values = [
                    int(mask) for mask in split_mask_slice.detach().cpu().tolist()
                ]
            else:
                split_mask_values = [int(mask) for mask in split_mask_slice]
            split_binary = self._create_split_binary_masks(
                split_mask_values,
                device,
                dtype=raw_leaf_embeddings.dtype,
            )
            select_f = raw_leaf_embeddings.new_zeros(edge_count, leaf_count)
            valid_leaf_columns = [
                (leaf_pos, bit)
                for leaf_pos, bit in enumerate(leaf_bits)
                if 0 <= bit < self.max_split_bits
            ]
            if valid_leaf_columns:
                target_columns = torch.tensor(
                    [leaf_pos for leaf_pos, _bit in valid_leaf_columns],
                    device=device,
                    dtype=torch.long,
                )
                source_columns = torch.tensor(
                    [bit for _leaf_pos, bit in valid_leaf_columns],
                    device=device,
                    dtype=torch.long,
                )
                select_f[:, target_columns] = split_binary.index_select(
                    1,
                    source_columns,
                )
            split_select = select_f > 0
            outside_f = (~split_select).to(dtype=raw_leaf_embeddings.dtype)
            valid_splits = torch.tensor(
                [split_mask != 0 for split_mask in split_mask_values],
                device=device,
                dtype=torch.bool,
            )

            inside_count = select_f.sum(dim=1)
            outside_count = outside_f.sum(dim=1)
            inside = select_f @ raw_leaf_embeddings
            outside = outside_f @ raw_leaf_embeddings
            inside = inside / inside_count.clamp_min(1.0).unsqueeze(1)
            outside = outside / outside_count.clamp_min(1.0).unsqueeze(1)
            inside = torch.where(
                inside_count.unsqueeze(1) > 0,
                inside,
                zero_raw.unsqueeze(0),
            )
            outside = torch.where(
                outside_count.unsqueeze(1) > 0,
                outside,
                zero_raw.unsqueeze(0),
            )

            clade_features = torch.cat([inside, outside], dim=1)
            projected = self.phyla_clade_proj(clade_features).to(dtype)
            context[b, edge_positions[:edge_count][valid_splits]] = projected[
                valid_splits
            ]
        return context

    def _compute_global_phyla_context(
        self,
        phyla_embeddings,
        leaf_idx_list,
        device,
        dtype,
    ):
        if not self.phyla_use_global_context or self.phyla_global_proj is None:
            return None
        pooled = []
        for b, leaf_indices in enumerate(leaf_idx_list):
            leaf_count = int(leaf_indices.numel())
            if leaf_count <= 0:
                pooled.append(
                    torch.zeros(
                        phyla_embeddings.size(-1),
                        device=device,
                        dtype=phyla_embeddings.dtype,
                    )
                )
                continue
            if phyla_embeddings.size(1) < leaf_count:
                raise ValueError(
                    f"Need {leaf_count} phyla embeddings, got {phyla_embeddings.size(1)}"
                )
            pooled.append(
                phyla_embeddings[b, :leaf_count].to(device=device).mean(dim=0)
            )
        pooled = torch.stack(pooled, dim=0).to(device=device)
        return self.phyla_global_proj(pooled).to(dtype)

    def _prepare_encoder_inputs(
        self,
        tokenized_tree_batch,
        t=None,
        phyla_embeddings=None,
    ):
        (
            padded_feature,
            padding_mask,
            padded_index,
            leaf_mask,
            leaf_idx,
            edge_mask,
            edge_split_masks,
        ) = tokenized_tree_batch

        x = padded_feature
        B, T_raw, D = x.shape

        phyla_embeddings = self._normalize_phyla_embeddings(phyla_embeddings, B)
        phyla_global_context = None
        phyla_clade_context = None
        if phyla_embeddings is not None:
            phyla_global_context = self._compute_global_phyla_context(
                phyla_embeddings,
                leaf_idx,
                x.device,
                x.dtype,
            )
            phyla_clade_context = self._compute_clade_phyla_token_context(
                phyla_embeddings,
                leaf_idx,
                edge_mask.bool(),
                edge_split_masks,
                T_raw,
                x.device,
                x.dtype,
            )
            if self.phyla_use_leaf_tokens:
                phyla_proj_full = self.phyla_proj(phyla_embeddings)
                x = x + (
                    self.phyla_leaf_scale
                    * self._compute_leaf_phyla_token_additions(
                        phyla_proj_full,
                        leaf_idx,
                        T_raw,
                        x.device,
                        x.dtype,
                    )
                )
            if self.phyla_use_split_tokens:
                x = x + (
                    self.phyla_split_scale
                    * self._compute_split_phyla_token_additions(
                        phyla_embeddings,
                        leaf_idx,
                        edge_mask.bool(),
                        edge_split_masks,
                        T_raw,
                        x.device,
                        x.dtype,
                    )
                )

        graph_token = self.graph_token.to(device=x.device, dtype=x.dtype).expand(B, 1, D)
        x = torch.cat([graph_token, x], dim=1)

        if padding_mask is not None:
            special_tokens_mask = torch.zeros(
                B, 1, dtype=padding_mask.dtype, device=padding_mask.device
            )
            padding_mask = torch.cat(
                [special_tokens_mask, padding_mask],
                dim=1,
            )

            if leaf_mask.dim() == 2:
                leaf_mask_special = torch.zeros(
                    B, 1, dtype=leaf_mask.dtype, device=leaf_mask.device
                )
                leaf_mask = torch.cat([leaf_mask_special, leaf_mask], dim=1)
            else:
                leaf_mask_expanded = torch.zeros(
                    B, T_raw, dtype=leaf_mask.dtype, device=leaf_mask.device
                )
                leaf_mask_expanded[0, : leaf_mask.size(0)] = leaf_mask
                leaf_mask_special = torch.zeros(
                    B, 1, dtype=leaf_mask.dtype, device=leaf_mask.device
                )
                leaf_mask = torch.cat([leaf_mask_special, leaf_mask_expanded], dim=1)

        if t is not None:
            time_sin_emb = self.create_sinusoidal_embedding(t, self.embed_dim)
            time_emb = self.time_embed(time_sin_emb)
            x = x + time_emb.unsqueeze(1)
        x = self.dropout(x)

        return (
            x,
            padding_mask,
            leaf_mask,
            leaf_idx,
            edge_mask,
            edge_split_masks,
            phyla_global_context,
            phyla_clade_context,
        )

    def _encode_with_layers(self, x, padding_mask, layers, final_layer_norm):
        for layer in layers:
            x, _ = layer(x, padding_mask=padding_mask)
        return final_layer_norm(x)

    def _coerce_time_batch(self, t, batch_size, device, dtype):
        if t is None:
            return None
        if torch.is_tensor(t):
            t_tensor = t.to(device=device, dtype=dtype).reshape(-1)
        elif isinstance(t, (list, tuple)):
            t_tensor = torch.tensor(
                [float(v) for v in t],
                device=device,
                dtype=dtype,
            ).reshape(-1)
        else:
            t_tensor = torch.tensor(
                [float(t)],
                device=device,
                dtype=dtype,
            )
        if t_tensor.numel() == 1 and int(batch_size) > 1:
            t_tensor = t_tensor.expand(int(batch_size))
        elif t_tensor.numel() != int(batch_size) and int(batch_size) > 0:
            t_tensor = t_tensor[:1].expand(int(batch_size))
        return t_tensor

    def _build_first_hit_head_input(self, padded_edges, *, t=None):
        if not self.first_hit_head_use_phase_input or self.first_hit_phase_proj is None:
            return padded_edges
        batch_size, max_edges, _ = padded_edges.shape
        if batch_size == 0 or max_edges == 0:
            return padded_edges
        phase_values = self._coerce_time_batch(
            t,
            batch_size=batch_size,
            device=padded_edges.device,
            dtype=padded_edges.dtype,
        )
        if phase_values is None:
            return padded_edges
        phase_features = self.first_hit_phase_proj(
            phase_values.unsqueeze(-1)
        ).unsqueeze(1)
        phase_features = phase_features.expand(batch_size, max_edges, self.embed_dim)
        return torch.cat([padded_edges, phase_features], dim=-1)

    def _masked_edge_mean(self, edge_values, edge_pad_mask):
        if edge_values.numel() == 0:
            return edge_values.new_zeros(edge_values.shape[0], edge_values.shape[-1])
        valid = (~edge_pad_mask).unsqueeze(-1).to(edge_values.dtype)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (edge_values * valid).sum(dim=1) / denom

    def _masked_edge_max(self, edge_values, edge_pad_mask):
        if edge_values.numel() == 0:
            return edge_values.new_zeros(edge_values.shape[0], edge_values.shape[-1])
        masked_values = edge_values.masked_fill(
            edge_pad_mask.unsqueeze(-1),
            float("-inf"),
        )
        max_values = masked_values.max(dim=1).values
        no_valid = edge_pad_mask.all(dim=1, keepdim=True)
        return torch.where(
            no_valid,
            edge_values.new_zeros(max_values.shape),
            max_values,
        )

    def _refine_first_hit_edges(self, padded_edges, edge_pad_mask):
        if self.first_hit_edge_refinement is None:
            return padded_edges
        if padded_edges.numel() == 0:
            return padded_edges
        refined = padded_edges.clone()
        batch_size = int(padded_edges.shape[0])
        for b in range(batch_size):
            valid_mask = ~edge_pad_mask[b]
            if not bool(valid_mask.any().item()):
                continue
            edge_values = refined[b, valid_mask]
            for layer in self.first_hit_edge_refinement:
                edge_values = layer(edge_values)
            refined[b, valid_mask] = edge_values
        return refined

    def _first_hit_autoregressive_encode_edges(self, edge_features):
        if (
            edge_features is None
            or self.first_hit_autoregressive_edge_encoder is None
        ):
            return None
        squeeze_batch = False
        if edge_features.ndim == 2:
            edge_features = edge_features.unsqueeze(0)
            squeeze_batch = True
        encoded = self.first_hit_autoregressive_edge_encoder(edge_features)
        if squeeze_batch:
            encoded = encoded.squeeze(0)
        return encoded

    def _first_hit_autoregressive_init_state(self, encoded_edges, available_mask=None):
        state = self.first_hit_autoregressive_start.to(
            device=encoded_edges.device,
            dtype=encoded_edges.dtype,
        )
        if encoded_edges.numel() == 0:
            return state
        if available_mask is None:
            available_mask = torch.ones(
                encoded_edges.shape[0],
                device=encoded_edges.device,
                dtype=torch.bool,
            )
        if bool(available_mask.any().item()):
            state = state + encoded_edges[available_mask].mean(dim=0)
        return state

    def _first_hit_autoregressive_step_logits(
        self,
        encoded_edges,
        state,
        available_mask,
    ):
        query = self.first_hit_autoregressive_query(state)
        keys = self.first_hit_autoregressive_key(encoded_edges)
        edge_logits = torch.matmul(keys, query.unsqueeze(-1)).squeeze(-1)
        edge_logits = edge_logits / math.sqrt(float(self.first_hit_head_hidden_dim))
        if edge_logits.numel() > 0:
            edge_logits = edge_logits.masked_fill(
                ~available_mask.bool(),
                float("-inf"),
            )
        stop_logit = self.first_hit_autoregressive_stop(state).reshape(1)
        return torch.cat([edge_logits, stop_logit], dim=0)

    def predict_first_hit_autoregressive_mask(
        self,
        edge_features,
        candidate_mask=None,
        max_steps=None,
    ):
        encoded_edges = self._first_hit_autoregressive_encode_edges(edge_features)
        if encoded_edges is None or encoded_edges.numel() == 0:
            device = (
                edge_features.device
                if edge_features is not None and hasattr(edge_features, "device")
                else None
            )
            return torch.zeros(0, dtype=torch.bool, device=device)
        n_edges = int(encoded_edges.shape[0])
        if candidate_mask is None:
            available_mask = torch.ones(
                n_edges,
                device=encoded_edges.device,
                dtype=torch.bool,
            )
        else:
            available_mask = candidate_mask.to(
                device=encoded_edges.device,
                dtype=torch.bool,
            )
        selected = torch.zeros(
            n_edges,
            device=encoded_edges.device,
            dtype=torch.bool,
        )
        if not bool(available_mask.any().item()):
            return selected
        state = self._first_hit_autoregressive_init_state(
            encoded_edges,
            available_mask=available_mask,
        )
        if max_steps is None or int(max_steps) <= 0:
            max_decode_steps = int(available_mask.sum().item()) + 1
        else:
            max_decode_steps = int(max_steps)
        for _ in range(max_decode_steps):
            step_available = available_mask & (~selected)
            logits = self._first_hit_autoregressive_step_logits(
                encoded_edges,
                state,
                step_available,
            )
            choice = int(torch.argmax(logits).item())
            if choice >= n_edges:
                break
            if not bool(step_available[choice].item()):
                break
            selected[choice] = True
            state = self.first_hit_autoregressive_update(
                encoded_edges[choice].unsqueeze(0),
                state.unsqueeze(0),
            ).squeeze(0)
        return selected

    def first_hit_autoregressive_group_loss(self, edge_features, target_mask):
        encoded_edges = self._first_hit_autoregressive_encode_edges(edge_features)
        if encoded_edges is None or encoded_edges.numel() == 0:
            zero = target_mask.new_zeros(())
            return zero, {
                "target_first_size": int(target_mask.sum().item()),
                "pred_first_size": 0,
                "recall": 0.0,
                "precision": 0.0,
                "jaccard": 0.0,
                "exact": 0.0,
            }
        target_mask = target_mask.to(device=encoded_edges.device, dtype=torch.bool)
        available_mask = torch.ones(
            encoded_edges.shape[0],
            device=encoded_edges.device,
            dtype=torch.bool,
        )
        state = self._first_hit_autoregressive_init_state(
            encoded_edges,
            available_mask=available_mask,
        )
        selected = torch.zeros_like(target_mask)
        losses = []
        while True:
            remaining_target = target_mask & (~selected)
            step_available = available_mask & (~selected)
            logits = self._first_hit_autoregressive_step_logits(
                encoded_edges,
                state,
                step_available,
            )
            if bool(remaining_target.any().item()):
                log_probs = torch.log_softmax(logits, dim=0)
                step_loss = -torch.logsumexp(log_probs[:-1][remaining_target], dim=0)
                losses.append(step_loss)
                candidate_scores = logits[:-1].masked_fill(
                    ~remaining_target,
                    float('-inf'),
                )
                chosen_idx = int(torch.argmax(candidate_scores).item())
                selected[chosen_idx] = True
                state = self.first_hit_autoregressive_update(
                    encoded_edges[chosen_idx].unsqueeze(0),
                    state.unsqueeze(0),
                ).squeeze(0)
                continue
            stop_target = torch.tensor(
                [int(encoded_edges.shape[0])],
                device=encoded_edges.device,
                dtype=torch.long,
            )
            losses.append(F.cross_entropy(logits.unsqueeze(0), stop_target))
            break
        loss = torch.stack(losses).mean()
        with torch.no_grad():
            pred_mask = self.predict_first_hit_autoregressive_mask(
                edge_features,
                candidate_mask=available_mask,
            )
            tp = (pred_mask & target_mask).sum().float()
            pred_n = pred_mask.sum().float()
            true_n = target_mask.sum().float()
            union = (pred_mask | target_mask).sum().float()
            exact = float(torch.equal(pred_mask, target_mask))
        stats = {
            "target_first_size": int(true_n.item()),
            "pred_first_size": int(pred_n.item()),
            "recall": float((tp / true_n.clamp_min(1.0)).item()) if int(true_n.item()) > 0 else 1.0,
            "precision": float((tp / pred_n.clamp_min(1.0)).item()) if int(pred_n.item()) > 0 else (1.0 if int(true_n.item()) == 0 else 0.0),
            "jaccard": float((tp / union.clamp_min(1.0)).item()) if int(union.item()) > 0 else 1.0,
            "exact": exact,
        }
        return loss, stats

    def _compute_first_hit_logits_from_edges(
        self,
        padded_edges,
        edge_pad_mask,
        *,
        t=None,
        graph_context=None,
        case_indices=None,
        topology_identity_embeddings=None,
        start_topology_features=None,
        start_topology_embeddings=None,
        start_topology_pad_mask=None,
        start_tree_graph_context=None,
        phyla_global_context=None,
    ):
        if self.first_hit_edge_refinement is not None:
            padded_edges = self._refine_first_hit_edges(padded_edges, edge_pad_mask)
        first_hit_head_input = self._build_first_hit_head_input(
            padded_edges,
            t=t,
        )
        if (
            self.phyla_use_global_context
            and phyla_global_context is not None
            and self.first_hit_phyla_global_proj is not None
        ):
            phyla_context = self.first_hit_phyla_global_proj(
                phyla_global_context.to(
                    device=first_hit_head_input.device,
                    dtype=first_hit_head_input.dtype,
                )
            )
            first_hit_head_input = first_hit_head_input + (
                self.phyla_global_context_scale * phyla_context.unsqueeze(1)
            )
        if self.first_hit_head_mode == "base":
            return self.first_hit_edge_head(first_hit_head_input)
        if self.first_hit_head_mode == "edge_refined_mlp":
            return self.first_hit_edge_head(first_hit_head_input)
        if self.first_hit_head_mode == "shared_mlp":
            hidden = self.first_hit_edge_head_shared(first_hit_head_input)
            return self.first_hit_edge_head_out(hidden)
        if self.first_hit_head_mode == "autoregressive_set":
            return self.first_hit_edge_head(first_hit_head_input)
        if self.first_hit_head_mode == "routed_adapter":
            hidden = self.first_hit_edge_head_shared(first_hit_head_input)
            pooled_input = self._masked_edge_mean(first_hit_head_input, edge_pad_mask)
            if graph_context is None:
                graph_context = padded_edges.new_zeros(
                    padded_edges.shape[0], self.embed_dim
                )
            router_input = torch.cat([pooled_input, graph_context], dim=-1)
            gamma_beta = self.first_hit_edge_head_router(router_input)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            hidden = hidden * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            return self.first_hit_edge_head_out(hidden)
        if self.first_hit_head_mode == "tree_conditioned_mlp":
            pooled_input = self._masked_edge_mean(first_hit_head_input, edge_pad_mask)
            if graph_context is None:
                graph_context = padded_edges.new_zeros(
                    padded_edges.shape[0], self.embed_dim
                )
            tree_context = self.first_hit_tree_context_proj(
                torch.cat([pooled_input, graph_context], dim=-1)
            )
            conditioned_input = torch.cat(
                [
                    first_hit_head_input,
                    tree_context.unsqueeze(1).expand(
                        -1, first_hit_head_input.shape[1], -1
                    ),
                ],
                dim=-1,
            )
            return self.first_hit_edge_head_conditioned(conditioned_input)
        if self.first_hit_head_mode == "case_adapted_mlp":
            if case_indices is None:
                raise ValueError("case_indices are required for case_adapted_mlp")
            case_indices = case_indices.to(device=padded_edges.device, dtype=torch.long)
            if case_indices.ndim != 1 or case_indices.shape[0] != padded_edges.shape[0]:
                raise ValueError(
                    "case_indices must have shape [batch] for case_adapted_mlp"
                )
            case_features = self.first_hit_case_embedding(case_indices)
            case_features = case_features.unsqueeze(1).expand(
                -1, first_hit_head_input.shape[1], -1
            )
            return self.first_hit_edge_head_adapted(
                torch.cat([first_hit_head_input, case_features], dim=-1)
            )
        if self.first_hit_head_mode == "frozen_start_case_mlp":
            if case_indices is None:
                raise ValueError("case_indices are required for frozen_start_case_mlp")
            case_indices = case_indices.to(device=padded_edges.device, dtype=torch.long)
            if case_indices.ndim != 1 or case_indices.shape[0] != padded_edges.shape[0]:
                raise ValueError(
                    "case_indices must have shape [batch] for frozen_start_case_mlp"
                )
            frozen_table = self.first_hit_frozen_start_case_embedding.to(
                device=padded_edges.device,
                dtype=padded_edges.dtype,
            )
            case_features = frozen_table.index_select(0, case_indices)
            if self.first_hit_frozen_start_case_proj is not None:
                case_features = self.first_hit_frozen_start_case_proj(case_features)
            case_features = case_features.unsqueeze(1).expand(
                -1, first_hit_head_input.shape[1], -1
            )
            return self.first_hit_edge_head_adapted(
                torch.cat([first_hit_head_input, case_features], dim=-1)
            )
        if self.first_hit_head_mode == "start_topology_adapter_mlp":
            if start_topology_features is None:
                raise ValueError(
                    "start_topology_features are required for start_topology_adapter_mlp"
                )
            start_topology_features = start_topology_features.to(
                device=padded_edges.device,
                dtype=padded_edges.dtype,
            )
            if (
                start_topology_features.ndim != 2
                or start_topology_features.shape[0] != padded_edges.shape[0]
                or start_topology_features.shape[1] != 3 * self.embed_dim
            ):
                raise ValueError(
                    "start_topology_features must have shape [batch, 3 * embed_dim] "
                    "for start_topology_adapter_mlp"
                )
            if graph_context is None:
                graph_context = padded_edges.new_zeros(
                    padded_edges.shape[0], self.embed_dim
                )
            topology_code = self.first_hit_start_topology_adapter(
                torch.cat([start_topology_features, graph_context], dim=-1)
            )
            topology_code = topology_code.unsqueeze(1).expand(
                -1, first_hit_head_input.shape[1], -1
            )
            return self.first_hit_edge_head_adapted(
                torch.cat([first_hit_head_input, topology_code], dim=-1)
            )
        if self.first_hit_head_mode == "start_topology_raw_pool_concat_mlp":
            if start_topology_features is None:
                raise ValueError(
                    "start_topology_features are required for "
                    "start_topology_raw_pool_concat_mlp"
                )
            start_topology_features = start_topology_features.to(
                device=padded_edges.device,
                dtype=padded_edges.dtype,
            )
            if (
                start_topology_features.ndim != 2
                or start_topology_features.shape[0] != padded_edges.shape[0]
                or start_topology_features.shape[1] != 3 * self.embed_dim
            ):
                raise ValueError(
                    "start_topology_features must have shape [batch, 3 * embed_dim] "
                    "for start_topology_raw_pool_concat_mlp"
                )
            topology_summary = start_topology_features.unsqueeze(1).expand(
                -1, first_hit_head_input.shape[1], -1
            )
            return self.first_hit_edge_head_start_topology_raw(
                torch.cat([first_hit_head_input, topology_summary], dim=-1)
            )
        if self.first_hit_head_mode == "start_tree_graph_token_mlp":
            if start_tree_graph_context is None:
                raise ValueError(
                    "start_tree_graph_context is required for start_tree_graph_token_mlp"
                )
            start_tree_graph_context = start_tree_graph_context.to(
                device=padded_edges.device,
                dtype=padded_edges.dtype,
            )
            if (
                start_tree_graph_context.ndim != 2
                or start_tree_graph_context.shape[0] != padded_edges.shape[0]
                or start_tree_graph_context.shape[1] != self.embed_dim
            ):
                raise ValueError(
                    "start_tree_graph_context must have shape [batch, embed_dim] "
                    "for start_tree_graph_token_mlp"
                )
            if self.first_hit_start_tree_graph_detach:
                start_tree_graph_context = start_tree_graph_context.detach()
            graph_code = start_tree_graph_context.unsqueeze(1).expand(
                -1, first_hit_head_input.shape[1], -1
            )
            return self.first_hit_edge_head_start_tree_graph(
                torch.cat([first_hit_head_input, graph_code], dim=-1)
            )
        if self.first_hit_head_mode == "topology_adapter_mlp":
            if topology_identity_embeddings is None:
                raise ValueError(
                    "topology_identity_embeddings are required for topology_adapter_mlp"
                )
            topology_mean = self._masked_edge_mean(
                topology_identity_embeddings,
                edge_pad_mask,
            )
            topology_max = self._masked_edge_max(
                topology_identity_embeddings,
                edge_pad_mask,
            )
            if graph_context is None:
                graph_context = padded_edges.new_zeros(
                    padded_edges.shape[0], self.embed_dim
                )
            topology_code = self.first_hit_topology_adapter(
                torch.cat([topology_mean, topology_max, graph_context], dim=-1)
            )
            topology_code = topology_code.unsqueeze(1).expand(
                -1, first_hit_head_input.shape[1], -1
            )
            return self.first_hit_edge_head_adapted(
                torch.cat([first_hit_head_input, topology_code], dim=-1)
            )
        if self.first_hit_head_mode == "topology_raw_pool_concat_mlp":
            if topology_identity_embeddings is None:
                raise ValueError(
                    "topology_identity_embeddings are required for topology_raw_pool_concat_mlp"
                )
            topology_mean = self._masked_edge_mean(
                topology_identity_embeddings,
                edge_pad_mask,
            )
            topology_max = self._masked_edge_max(
                topology_identity_embeddings,
                edge_pad_mask,
            )
            topology_summary = torch.cat([topology_mean, topology_max], dim=-1)
            topology_summary = topology_summary.unsqueeze(1).expand(
                -1, first_hit_head_input.shape[1], -1
            )
            return self.first_hit_edge_head_raw_topology(
                torch.cat([first_hit_head_input, topology_summary], dim=-1)
            )
        if self.first_hit_head_mode == "topology_attention_adapter_mlp":
            if topology_identity_embeddings is None:
                raise ValueError(
                    "topology_identity_embeddings are required for topology_attention_adapter_mlp"
                )
            pooled_input = self._masked_edge_mean(first_hit_head_input, edge_pad_mask)
            if graph_context is None:
                graph_context = padded_edges.new_zeros(
                    padded_edges.shape[0], self.embed_dim
                )
            topology_query = self.first_hit_topology_query(
                torch.cat([pooled_input, graph_context], dim=-1)
            )
            topology_keys = self.first_hit_topology_key(topology_identity_embeddings)
            topology_values = self.first_hit_topology_value(topology_identity_embeddings)
            attention_scores = (
                torch.sum(topology_keys * topology_query.unsqueeze(1), dim=-1)
                / math.sqrt(float(self.first_hit_head_case_dim))
            )
            attention_scores = attention_scores.masked_fill(
                edge_pad_mask,
                float("-inf"),
            )
            attention_weights = torch.softmax(attention_scores, dim=1)
            no_valid = edge_pad_mask.all(dim=1, keepdim=True)
            attention_weights = torch.where(
                no_valid,
                attention_weights.new_zeros(attention_weights.shape),
                attention_weights,
            )
            attention_summary = torch.sum(
                topology_values * attention_weights.unsqueeze(-1),
                dim=1,
            )
            topology_mean = self._masked_edge_mean(
                topology_values,
                edge_pad_mask,
            )
            topology_max = self._masked_edge_max(
                topology_values,
                edge_pad_mask,
            )
            topology_code = self.first_hit_topology_adapter(
                torch.cat(
                    [
                        topology_query,
                        attention_summary,
                        topology_mean,
                        topology_max,
                    ],
                    dim=-1,
                )
            )
            topology_code = topology_code.unsqueeze(1).expand(
                -1, first_hit_head_input.shape[1], -1
            )
            return self.first_hit_edge_head_adapted(
                torch.cat([first_hit_head_input, topology_code], dim=-1)
            )
        if self.first_hit_head_mode == "topology_cross_attn_mlp":
            if topology_identity_embeddings is None:
                raise ValueError(
                    "topology_identity_embeddings are required for topology_cross_attn_mlp"
                )
            if graph_context is None:
                graph_context = padded_edges.new_zeros(
                    padded_edges.shape[0], self.embed_dim
                )
            graph_context_expanded = graph_context.unsqueeze(1).expand(
                -1, first_hit_head_input.shape[1], -1
            )
            cross_query = self.first_hit_topology_cross_query(
                torch.cat([first_hit_head_input, graph_context_expanded], dim=-1)
            )
            topology_keys = self.first_hit_topology_cross_key(
                topology_identity_embeddings
            )
            topology_values = self.first_hit_topology_cross_value(
                topology_identity_embeddings
            )
            attention_scores = torch.matmul(
                cross_query,
                topology_keys.transpose(1, 2),
            ) / math.sqrt(float(self.first_hit_head_hidden_dim))
            attention_scores = attention_scores.masked_fill(
                edge_pad_mask.unsqueeze(1),
                float("-inf"),
            )
            attention_weights = torch.softmax(attention_scores, dim=-1)
            no_valid = edge_pad_mask.all(dim=1, keepdim=True).unsqueeze(-1)
            attention_weights = torch.where(
                no_valid,
                attention_weights.new_zeros(attention_weights.shape),
                attention_weights,
            )
            topology_context = torch.matmul(attention_weights, topology_values)
            return self.first_hit_edge_head_cross_attn(
                torch.cat(
                    [
                        first_hit_head_input,
                        graph_context_expanded,
                        topology_context,
                    ],
                    dim=-1,
                )
            )
        if self.first_hit_head_mode == "start_topology_cross_attn_mlp":
            if start_topology_embeddings is None or start_topology_pad_mask is None:
                raise ValueError(
                    "start_topology_embeddings and start_topology_pad_mask are "
                    "required for start_topology_cross_attn_mlp"
                )
            start_topology_embeddings = start_topology_embeddings.to(
                device=padded_edges.device,
                dtype=padded_edges.dtype,
            )
            start_topology_pad_mask = start_topology_pad_mask.to(
                device=padded_edges.device,
                dtype=torch.bool,
            )
            if graph_context is None:
                graph_context = padded_edges.new_zeros(
                    padded_edges.shape[0], self.embed_dim
                )
            graph_context_expanded = graph_context.unsqueeze(1).expand(
                -1, first_hit_head_input.shape[1], -1
            )
            cross_query = self.first_hit_topology_cross_query(
                torch.cat([first_hit_head_input, graph_context_expanded], dim=-1)
            )
            topology_keys = self.first_hit_topology_cross_key(
                start_topology_embeddings
            )
            topology_values = self.first_hit_topology_cross_value(
                start_topology_embeddings
            )
            attention_scores = torch.matmul(
                cross_query,
                topology_keys.transpose(1, 2),
            ) / math.sqrt(float(self.first_hit_head_hidden_dim))
            attention_scores = attention_scores.masked_fill(
                start_topology_pad_mask.unsqueeze(1),
                float("-inf"),
            )
            attention_weights = torch.softmax(attention_scores, dim=-1)
            no_valid = start_topology_pad_mask.all(dim=1, keepdim=True).unsqueeze(-1)
            attention_weights = torch.where(
                no_valid,
                attention_weights.new_zeros(attention_weights.shape),
                attention_weights,
            )
            topology_context = torch.matmul(attention_weights, topology_values)
            return self.first_hit_edge_head_cross_attn(
                torch.cat(
                    [
                        first_hit_head_input,
                        graph_context_expanded,
                        topology_context,
                    ],
                    dim=-1,
                )
            )
        raise ValueError(f"Unknown first_hit_head_mode: {self.first_hit_head_mode}")

    def _decode_outputs(
        self,
        x,
        leaf_mask,
        leaf_idx,
        edge_mask,
        edge_split_masks,
        t=None,
        return_all_tokens=True,
        return_leafs_only=False,
        return_edges_only=False,
        return_edge_features=False,
        return_first_hit_logits=False,
        return_boundary_vanish_logits=False,
        autoregressive=False,
        autoregressive_component_groups=None,
        autoregressive_case_indices=None,
        autoregressive_start_topology_features=None,
        autoregressive_structured_subset_top_member_pairs=None,
        first_hit_case_indices=None,
        first_hit_start_topology_features=None,
        first_hit_start_topology_embeddings=None,
        first_hit_start_topology_pad_mask=None,
        first_hit_start_tree_graph_context=None,
        phyla_global_context=None,
        phyla_clade_context=None,
    ):
        B, _T, D = x.shape
        leaf_idx_list = list(leaf_idx) if isinstance(leaf_idx, (list, tuple)) else [leaf_idx]
        is_single_tree = B == 1 and not isinstance(leaf_idx, (list, tuple))

        if return_leafs_only:
            if is_single_tree and B == 1:
                if len(leaf_idx_list[0]) > 0:
                    adjusted_indices = leaf_idx_list[0] + 1
                    valid_mask = adjusted_indices < x.size(1)
                    valid_indices = adjusted_indices[valid_mask]
                    if valid_indices.numel() > 0:
                        return x[0, valid_indices].unsqueeze(0)
                    return torch.zeros(1, 0, D, device=x.device)
                return torch.zeros(1, 0, D, device=x.device)

            batch_leaf_outputs = []
            for b in range(B):
                if len(leaf_idx_list[b]) > 0:
                    adjusted_indices = leaf_idx_list[b] + 1
                    valid_mask = adjusted_indices < x.size(1)
                    valid_indices = adjusted_indices[valid_mask]
                    if valid_indices.numel() > 0:
                        batch_leaf_outputs.append(x[b, valid_indices])
                    else:
                        batch_leaf_outputs.append(torch.zeros(0, D, device=x.device))
                else:
                    batch_leaf_outputs.append(torch.zeros(0, D, device=x.device))

            if batch_leaf_outputs:
                max_leaf_len = max(out.size(0) for out in batch_leaf_outputs)
                if max_leaf_len > 0:
                    padded_leaf_outputs = torch.zeros(
                        B, max_leaf_len, D, device=x.device
                    )
                    for b, out in enumerate(batch_leaf_outputs):
                        if out.size(0) > 0:
                            padded_leaf_outputs[b, : out.size(0)] = out
                    return padded_leaf_outputs
            return torch.zeros(B, 0, D, device=x.device)

        if autoregressive:
            x_no_graph = x[:, 1:, :]
            all_group_logits = []
            num_leaves = [int(leaf_mask[b].sum().item()) + 1 for b in range(B)]

            if autoregressive_component_groups is None:
                batch_polytomy_index, batch_polytomy_splits = (
                    get_batch_structural_polytomy_indices(
                        edge_split_masks,
                        edge_mask,
                        min_children=3,
                        num_leaves=num_leaves,
                    )
                )
            else:
                batch_polytomy_index, batch_polytomy_splits = (
                    get_batch_explicit_structural_group_indices(
                        edge_split_masks,
                        edge_mask,
                        autoregressive_component_groups,
                        num_leaves=num_leaves,
                    )
                )

            autoregressive_case_context = None
            if self.autoregressive_use_case_conditioning:
                if autoregressive_case_indices is None:
                    raise ValueError(
                        "autoregressive_case_indices are required when "
                        "autoregressive_use_case_conditioning is enabled"
                    )
                autoregressive_case_indices = autoregressive_case_indices.to(
                    device=x.device,
                    dtype=torch.long,
                )
                if (
                    autoregressive_case_indices.ndim != 1
                    or autoregressive_case_indices.shape[0] != B
                ):
                    raise ValueError(
                        "autoregressive_case_indices must have shape [batch]"
                    )
                autoregressive_case_context = self.autoregressive_case_proj(
                    self.autoregressive_case_embedding(autoregressive_case_indices)
                )

            autoregressive_start_topology_context = None
            autoregressive_start_topology_head_context = None
            autoregressive_phyla_context = None
            if (
                self.phyla_use_global_context
                and phyla_global_context is not None
                and self.autoregressive_phyla_global_proj is not None
            ):
                autoregressive_phyla_context = self.autoregressive_phyla_global_proj(
                    phyla_global_context.to(device=x.device, dtype=x.dtype)
                )
            if self.autoregressive_use_start_topology_conditioning:
                if (
                    self.autoregressive_start_topology_conditioning_mode
                    in {"frozen_case_probe", "frozen_case_probe_additive"}
                ):
                    if autoregressive_case_indices is None:
                        raise ValueError(
                            "autoregressive_case_indices are required for "
                            "frozen case-probe conditioning"
                        )
                    autoregressive_case_indices = autoregressive_case_indices.to(
                        device=x.device,
                        dtype=torch.long,
                    )
                    if (
                        autoregressive_case_indices.ndim != 1
                        or autoregressive_case_indices.shape[0] != B
                    ):
                        raise ValueError(
                            "autoregressive_case_indices must have shape [batch] "
                            "for frozen case-probe conditioning"
                        )
                    frozen_table = self.autoregressive_frozen_start_case_embedding.to(
                        device=x.device,
                        dtype=x.dtype,
                    )
                    frozen_context = frozen_table.index_select(
                        0,
                        autoregressive_case_indices,
                    )
                    if (
                        self.autoregressive_start_topology_conditioning_mode
                        == "frozen_case_probe_additive"
                    ):
                        autoregressive_start_topology_context = (
                            self.autoregressive_frozen_start_case_proj(frozen_context)
                        )
                    else:
                        autoregressive_start_topology_head_context = frozen_context
                else:
                    if autoregressive_start_topology_features is None:
                        raise ValueError(
                            "autoregressive_start_topology_features are required when "
                            "autoregressive_use_start_topology_conditioning is enabled"
                        )
                    autoregressive_start_topology_features = (
                        autoregressive_start_topology_features.to(
                            device=x.device,
                            dtype=x.dtype,
                        )
                    )
                    if (
                        autoregressive_start_topology_features.ndim != 2
                        or autoregressive_start_topology_features.shape[0] != B
                        or autoregressive_start_topology_features.shape[1]
                        != 3 * self.embed_dim
                    ):
                        raise ValueError(
                            "autoregressive_start_topology_features must have shape "
                            "[batch, 3 * embed_dim]"
                        )
                    if (
                        self.autoregressive_start_topology_conditioning_mode
                        == "additive"
                    ):
                        autoregressive_start_topology_context = (
                            self.autoregressive_start_topology_adapter(
                                autoregressive_start_topology_features
                            )
                        )
                    else:
                        autoregressive_start_topology_head_context = (
                            self.autoregressive_start_topology_head_proj(
                                autoregressive_start_topology_features
                            )
                        )

            for b, groups in enumerate(batch_polytomy_index):
                for num, group in enumerate(groups):
                    if group.size(0) <= 1:
                        continue
                    group_embeddings = x_no_graph[b, group, :]
                    group_splits = batch_polytomy_splits[b][num]
                    group_identity = self.create_split_identity_embedding(
                        group_splits, x.device
                    )

                    group_embeddings = group_embeddings + (
                        self.split_identity_scale * group_identity
                    )
                    if autoregressive_case_context is not None:
                        group_embeddings = group_embeddings + autoregressive_case_context[
                            b
                        ].unsqueeze(0)
                    if autoregressive_start_topology_context is not None:
                        group_embeddings = (
                            group_embeddings
                            + autoregressive_start_topology_context[b].unsqueeze(0)
                        )
                    if autoregressive_phyla_context is not None:
                        group_embeddings = group_embeddings + (
                            self.phyla_global_context_scale
                            * autoregressive_phyla_context[b].unsqueeze(0)
                        )
                    if phyla_clade_context is not None:
                        group_embeddings = group_embeddings + (
                            self.phyla_clade_context_scale
                            * phyla_clade_context[b, group, :].to(
                                device=group_embeddings.device,
                                dtype=group_embeddings.dtype,
                            )
                        )
                    for refinement_block in self.autoregressive_group_refinement:
                        group_embeddings = refinement_block(group_embeddings)
                    group_head_context = (
                        None
                        if autoregressive_start_topology_head_context is None
                        else autoregressive_start_topology_head_context[b]
                    )
                    if self.autoregressive_head_mode == "structured_subset":
                        head_outputs = self.structured_subset_head(
                            group_embeddings,
                            context=group_head_context,
                            top_member_pairs=autoregressive_structured_subset_top_member_pairs,
                        )
                        logits = head_outputs["logits"]
                    else:
                        head_outputs = {}
                        logits = self.pairwise_head(
                            group_embeddings,
                            context=group_head_context,
                        )

                    group_output = {
                        "batch_index": b,
                        "group_indices": group,
                        "polytomy_pred": self.group_head(
                            self.final_layer_norm(group_embeddings).mean(dim=0)
                        ),
                        "logits": logits,
                        "splits_represented": group_splits,
                        "group_embeddings": group_embeddings,
                        "decoder_mode": self.autoregressive_head_mode,
                    }
                    if (
                        self.autoregressive_head_mode == "structured_subset"
                        and autoregressive_structured_subset_top_member_pairs is not None
                    ):
                        group_output["_structured_subset_head"] = self.structured_subset_head
                        group_output["_structured_subset_context"] = group_head_context
                        group_output["_structured_subset_top_member_pairs"] = int(
                            autoregressive_structured_subset_top_member_pairs
                        )
                    group_output.update(head_outputs)
                    all_group_logits.append(group_output)
            return all_group_logits

        if return_edges_only:
            x_no_graph = x[:, 1:, :]
            edge_mask_bool = edge_mask.bool()
            edge_lists = [x_no_graph[b][edge_mask_bool[b]] for b in range(B)]
            edge_clade_context_lists = None
            if phyla_clade_context is not None:
                edge_clade_context_lists = [
                    phyla_clade_context[b][edge_mask_bool[b]] for b in range(B)
                ]
            split_identity_lists = []
            for b in range(B):
                n_b = edge_lists[b].size(0)
                if n_b == 0:
                    split_identity_lists.append(torch.zeros(0, D, device=x.device, dtype=x.dtype))
                    continue
                split_identity_lists.append(
                    self.create_split_identity_embedding(
                        edge_split_masks[b][:n_b],
                        x.device,
                    ).to(dtype=x.dtype)
                )
            max_edges = max((e.size(0) for e in edge_lists), default=0)

            if max_edges == 0:
                return torch.zeros(B, 0, D, device=x.device), torch.ones(
                    B, 0, device=x.device, dtype=torch.bool
                )

            padded_edges = torch.zeros(B, max_edges, D, device=x.device, dtype=x.dtype)
            padded_split_identity = torch.zeros(
                B, max_edges, D, device=x.device, dtype=x.dtype
            )
            padded_clade_context = None
            if edge_clade_context_lists is not None:
                padded_clade_context = torch.zeros(
                    B, max_edges, D, device=x.device, dtype=x.dtype
                )
            edge_pad_mask = torch.ones(B, max_edges, device=x.device, dtype=torch.bool)

            for b, edges_b in enumerate(edge_lists):
                n_b = edges_b.size(0)
                if n_b == 0:
                    continue
                split_identity = split_identity_lists[b]
                padded_edges[b, :n_b] = edges_b + (
                    self.split_identity_scale * split_identity
                )
                padded_split_identity[b, :n_b] = split_identity
                if padded_clade_context is not None:
                    padded_clade_context[b, :n_b] = edge_clade_context_lists[b][
                        :n_b
                    ].to(device=x.device, dtype=x.dtype)
                edge_pad_mask[b, :n_b] = False

            edge_outputs = self.edge_output_layer(padded_edges)
            if return_first_hit_logits or return_boundary_vanish_logits:
                first_hit_logits = None
                boundary_vanish_logits = None
                if return_first_hit_logits:
                    first_hit_edges = padded_edges
                    if padded_clade_context is not None:
                        first_hit_edges = first_hit_edges + (
                            self.phyla_clade_context_scale * padded_clade_context
                        )
                    first_hit_logits = self._compute_first_hit_logits_from_edges(
                        first_hit_edges,
                        edge_pad_mask,
                        t=t,
                        graph_context=x[:, 0, :],
                        case_indices=first_hit_case_indices,
                        topology_identity_embeddings=padded_split_identity,
                        start_topology_features=first_hit_start_topology_features,
                        start_topology_embeddings=first_hit_start_topology_embeddings,
                        start_topology_pad_mask=first_hit_start_topology_pad_mask,
                        start_tree_graph_context=first_hit_start_tree_graph_context,
                        phyla_global_context=phyla_global_context,
                    )
                if return_boundary_vanish_logits:
                    boundary_vanish_logits = self.boundary_vanish_edge_head(
                        padded_edges
                    )
                if return_first_hit_logits and return_boundary_vanish_logits:
                    if return_edge_features:
                        return (
                            edge_outputs,
                            edge_pad_mask,
                            padded_edges,
                            first_hit_logits,
                            boundary_vanish_logits,
                        )
                    return (
                        edge_outputs,
                        edge_pad_mask,
                        first_hit_logits,
                        boundary_vanish_logits,
                    )
                if return_first_hit_logits:
                    if return_edge_features:
                        return edge_outputs, edge_pad_mask, padded_edges, first_hit_logits
                    return edge_outputs, edge_pad_mask, first_hit_logits
                if return_edge_features:
                    return edge_outputs, edge_pad_mask, padded_edges, boundary_vanish_logits
                return edge_outputs, edge_pad_mask, boundary_vanish_logits

            if return_edge_features:
                return edge_outputs, edge_pad_mask, padded_edges
            return edge_outputs, edge_pad_mask

        if return_all_tokens:
            return x
        return self.output_layer(x[:, 0])

    def forward(
        self,
        tokenized_tree_batch,
        t=None,
        phyla_embeddings=None,
        return_all_tokens=True,
        return_leafs_only=False,
        return_edges_only=False,
        return_edge_features=False,
        return_first_hit_logits=False,
        return_boundary_vanish_logits=False,
        autoregressive=False,
        autoregressive_component_groups=None,
        autoregressive_case_indices=None,
        autoregressive_start_topology_features=None,
        autoregressive_structured_subset_top_member_pairs=None,
        first_hit_case_indices=None,
        first_hit_start_topology_features=None,
        first_hit_start_topology_embeddings=None,
        first_hit_start_topology_pad_mask=None,
        first_hit_start_tree_graph_context=None,
    ):
        (
            x,
            padding_mask,
            leaf_mask,
            leaf_idx,
            edge_mask,
            edge_split_masks,
            phyla_global_context,
            phyla_clade_context,
        ) = (
            self._prepare_encoder_inputs(
                tokenized_tree_batch,
                t=t,
                phyla_embeddings=phyla_embeddings,
            )
        )
        x = self._encode_with_layers(
            x,
            padding_mask=padding_mask,
            layers=self.layers,
            final_layer_norm=self.final_layer_norm,
        )
        return self._decode_outputs(
            x,
            leaf_mask=leaf_mask,
            leaf_idx=leaf_idx,
            edge_mask=edge_mask,
            edge_split_masks=edge_split_masks,
            t=t,
            return_all_tokens=return_all_tokens,
            return_leafs_only=return_leafs_only,
            return_edges_only=return_edges_only,
            return_edge_features=return_edge_features,
            return_first_hit_logits=return_first_hit_logits,
            return_boundary_vanish_logits=return_boundary_vanish_logits,
            autoregressive=autoregressive,
            autoregressive_component_groups=autoregressive_component_groups,
            autoregressive_case_indices=autoregressive_case_indices,
            autoregressive_start_topology_features=autoregressive_start_topology_features,
            autoregressive_structured_subset_top_member_pairs=autoregressive_structured_subset_top_member_pairs,
            first_hit_case_indices=first_hit_case_indices,
            first_hit_start_topology_features=first_hit_start_topology_features,
            first_hit_start_topology_embeddings=first_hit_start_topology_embeddings,
            first_hit_start_topology_pad_mask=first_hit_start_topology_pad_mask,
            first_hit_start_tree_graph_context=first_hit_start_tree_graph_context,
            phyla_global_context=phyla_global_context,
            phyla_clade_context=phyla_clade_context,
        )


class TreeDenoiserTokenGTTwoBlock(TreeDenoiserTokenGT):
    def __init__(
        self,
        *args,
        block2_n_layers=None,
        block2_input_mode="direct",
        block2_drop_path_rate=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.block2_n_layers = int(
            self.n_layers if block2_n_layers is None else block2_n_layers
        )
        self.block2_input_mode = str(block2_input_mode)
        block2_drop_path_rate = (
            self.block2_n_layers > 1 and kwargs.get("drop_path_rate", 0.0)
            if block2_drop_path_rate is None
            else float(block2_drop_path_rate)
        )
        dprates = [
            block2_drop_path_rate * i / (self.block2_n_layers - 1)
            if self.block2_n_layers > 1
            else 0.0
            for i in range(self.block2_n_layers)
        ]
        self.block2_layers = nn.ModuleList(
            [
                TreeGraphEncoderLayer(
                    embed_dim=self.embed_dim,
                    ffn_dim=self.embed_dim * 4,
                    n_heads=kwargs["n_heads"],
                    dropout=kwargs["dropout"],
                    attention_dropout=kwargs["attention_dropout"],
                    activation_dropout=kwargs["activation_dropout"],
                    drop_path=dprates[i],
                    use_performer=kwargs["use_performer"],
                    performer_nb_features=kwargs["performer_nb_features"],
                    performer_generalized_attention=kwargs[
                        "performer_generalized_attention"
                    ],
                    layernorm_style=kwargs["layernorm_style"],
                    n_layers=self.block2_n_layers,
                )
                for i in range(self.block2_n_layers)
            ]
        )
        self.block2_final_layer_norm = nn.LayerNorm(self.embed_dim)
        if self.block2_input_mode == "direct":
            self.block2_bridge = nn.Identity()
        elif self.block2_input_mode == "residual_mlp":
            self.block2_bridge = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Dropout(kwargs["dropout"]),
            )
        elif self.block2_input_mode == "concat_raw_mlp":
            self.block2_bridge = nn.Sequential(
                nn.LayerNorm(2 * self.embed_dim),
                nn.Linear(2 * self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Dropout(kwargs["dropout"]),
            )
        else:
            raise ValueError(
                f"Unknown block2_input_mode={self.block2_input_mode!r}"
            )
        self.apply(lambda m: init_params(m, self.n_layers + self.block2_n_layers))

    def _build_block2_input(self, stage1_x, encoder_input_x):
        if self.block2_input_mode == "direct":
            return stage1_x
        if self.block2_input_mode == "residual_mlp":
            return stage1_x + self.block2_bridge(stage1_x)
        return self.block2_bridge(torch.cat([stage1_x, encoder_input_x], dim=-1))

    def forward(
        self,
        tokenized_tree_batch,
        t=None,
        phyla_embeddings=None,
        return_all_tokens=True,
        return_leafs_only=False,
        return_edges_only=False,
        return_edge_features=False,
        return_first_hit_logits=False,
        return_boundary_vanish_logits=False,
        autoregressive=False,
        autoregressive_component_groups=None,
        autoregressive_case_indices=None,
        autoregressive_start_topology_features=None,
        autoregressive_structured_subset_top_member_pairs=None,
        first_hit_case_indices=None,
        first_hit_start_topology_features=None,
        first_hit_start_topology_embeddings=None,
        first_hit_start_topology_pad_mask=None,
        first_hit_start_tree_graph_context=None,
    ):
        (
            x,
            padding_mask,
            leaf_mask,
            leaf_idx,
            edge_mask,
            edge_split_masks,
            phyla_global_context,
            phyla_clade_context,
        ) = (
            self._prepare_encoder_inputs(
                tokenized_tree_batch,
                t=t,
                phyla_embeddings=phyla_embeddings,
            )
        )
        stage1_x = self._encode_with_layers(
            x,
            padding_mask=padding_mask,
            layers=self.layers,
            final_layer_norm=self.final_layer_norm,
        )
        block2_x = self._build_block2_input(stage1_x, x)
        x = self._encode_with_layers(
            block2_x,
            padding_mask=padding_mask,
            layers=self.block2_layers,
            final_layer_norm=self.block2_final_layer_norm,
        )
        return self._decode_outputs(
            x,
            leaf_mask=leaf_mask,
            leaf_idx=leaf_idx,
            edge_mask=edge_mask,
            edge_split_masks=edge_split_masks,
            t=t,
            return_all_tokens=return_all_tokens,
            return_leafs_only=return_leafs_only,
            return_edges_only=return_edges_only,
            return_edge_features=return_edge_features,
            return_first_hit_logits=return_first_hit_logits,
            return_boundary_vanish_logits=return_boundary_vanish_logits,
            autoregressive=autoregressive,
            autoregressive_component_groups=autoregressive_component_groups,
            autoregressive_case_indices=autoregressive_case_indices,
            autoregressive_start_topology_features=autoregressive_start_topology_features,
            autoregressive_structured_subset_top_member_pairs=autoregressive_structured_subset_top_member_pairs,
            first_hit_case_indices=first_hit_case_indices,
            first_hit_start_topology_features=first_hit_start_topology_features,
            first_hit_start_topology_embeddings=first_hit_start_topology_embeddings,
            first_hit_start_topology_pad_mask=first_hit_start_topology_pad_mask,
            first_hit_start_tree_graph_context=first_hit_start_tree_graph_context,
            phyla_global_context=phyla_global_context,
            phyla_clade_context=phyla_clade_context,
        )


class TreeDenoiserTokenGTMultiBlock(TreeDenoiserTokenGT):
    def __init__(
        self,
        *args,
        num_model_blocks=3,
        refine_block_n_layers=None,
        refine_block_input_mode="direct",
        refine_block_drop_path_rate=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_model_blocks = int(num_model_blocks)
        if self.num_model_blocks < 2:
            raise ValueError("num_model_blocks must be at least 2.")
        self.refine_block_n_layers = int(
            self.n_layers if refine_block_n_layers is None else refine_block_n_layers
        )
        self.refine_block_input_mode = str(refine_block_input_mode)
        refine_block_drop_path_rate = (
            kwargs.get("drop_path_rate", 0.0)
            if refine_block_drop_path_rate is None
            else float(refine_block_drop_path_rate)
        )
        self.refine_blocks = nn.ModuleList()
        for _ in range(self.num_model_blocks - 1):
            dprates = [
                refine_block_drop_path_rate * i / (self.refine_block_n_layers - 1)
                if self.refine_block_n_layers > 1
                else 0.0
                for i in range(self.refine_block_n_layers)
            ]
            layers = nn.ModuleList(
                [
                    TreeGraphEncoderLayer(
                        embed_dim=self.embed_dim,
                        ffn_dim=self.embed_dim * 4,
                        n_heads=kwargs["n_heads"],
                        dropout=kwargs["dropout"],
                        attention_dropout=kwargs["attention_dropout"],
                        activation_dropout=kwargs["activation_dropout"],
                        drop_path=dprates[i],
                        use_performer=kwargs["use_performer"],
                        performer_nb_features=kwargs["performer_nb_features"],
                        performer_generalized_attention=kwargs[
                            "performer_generalized_attention"
                        ],
                        layernorm_style=kwargs["layernorm_style"],
                        n_layers=self.refine_block_n_layers,
                    )
                    for i in range(self.refine_block_n_layers)
                ]
            )
            final_norm = nn.LayerNorm(self.embed_dim)
            if self.refine_block_input_mode == "direct":
                bridge = nn.Identity()
            elif self.refine_block_input_mode == "residual_mlp":
                bridge = nn.Sequential(
                    nn.LayerNorm(self.embed_dim),
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.GELU(),
                    nn.Dropout(kwargs["dropout"]),
                )
            elif self.refine_block_input_mode == "concat_raw_mlp":
                bridge = nn.Sequential(
                    nn.LayerNorm(2 * self.embed_dim),
                    nn.Linear(2 * self.embed_dim, self.embed_dim),
                    nn.GELU(),
                    nn.Dropout(kwargs["dropout"]),
                )
            else:
                raise ValueError(
                    f"Unknown refine_block_input_mode={self.refine_block_input_mode!r}"
                )
            self.refine_blocks.append(
                nn.ModuleDict(
                    {
                        "layers": layers,
                        "final_norm": final_norm,
                        "bridge": bridge,
                    }
                )
            )
        self.apply(
            lambda m: init_params(
                m,
                self.n_layers
                + ((self.num_model_blocks - 1) * self.refine_block_n_layers),
            )
        )

    def _build_refine_block_input(self, prev_x, encoder_input_x, bridge):
        if self.refine_block_input_mode == "direct":
            return prev_x
        if self.refine_block_input_mode == "residual_mlp":
            return prev_x + bridge(prev_x)
        return bridge(torch.cat([prev_x, encoder_input_x], dim=-1))

    def forward(
        self,
        tokenized_tree_batch,
        t=None,
        phyla_embeddings=None,
        return_all_tokens=True,
        return_leafs_only=False,
        return_edges_only=False,
        return_edge_features=False,
        return_first_hit_logits=False,
        return_boundary_vanish_logits=False,
        autoregressive=False,
        autoregressive_component_groups=None,
        autoregressive_case_indices=None,
        autoregressive_start_topology_features=None,
        autoregressive_structured_subset_top_member_pairs=None,
        first_hit_case_indices=None,
        first_hit_start_topology_features=None,
        first_hit_start_topology_embeddings=None,
        first_hit_start_topology_pad_mask=None,
        first_hit_start_tree_graph_context=None,
    ):
        (
            x,
            padding_mask,
            leaf_mask,
            leaf_idx,
            edge_mask,
            edge_split_masks,
            phyla_global_context,
            phyla_clade_context,
        ) = (
            self._prepare_encoder_inputs(
                tokenized_tree_batch,
                t=t,
                phyla_embeddings=phyla_embeddings,
            )
        )
        encoder_input_x = x
        x = self._encode_with_layers(
            x,
            padding_mask=padding_mask,
            layers=self.layers,
            final_layer_norm=self.final_layer_norm,
        )
        for block in self.refine_blocks:
            block_input_x = self._build_refine_block_input(
                x,
                encoder_input_x,
                block["bridge"],
            )
            x = self._encode_with_layers(
                block_input_x,
                padding_mask=padding_mask,
                layers=block["layers"],
                final_layer_norm=block["final_norm"],
            )
        return self._decode_outputs(
            x,
            leaf_mask=leaf_mask,
            leaf_idx=leaf_idx,
            edge_mask=edge_mask,
            edge_split_masks=edge_split_masks,
            t=t,
            return_all_tokens=return_all_tokens,
            return_leafs_only=return_leafs_only,
            return_edges_only=return_edges_only,
            return_edge_features=return_edge_features,
            return_first_hit_logits=return_first_hit_logits,
            return_boundary_vanish_logits=return_boundary_vanish_logits,
            autoregressive=autoregressive,
            autoregressive_component_groups=autoregressive_component_groups,
            autoregressive_case_indices=autoregressive_case_indices,
            autoregressive_start_topology_features=autoregressive_start_topology_features,
            autoregressive_structured_subset_top_member_pairs=autoregressive_structured_subset_top_member_pairs,
            first_hit_case_indices=first_hit_case_indices,
            first_hit_start_topology_features=first_hit_start_topology_features,
            first_hit_start_topology_embeddings=first_hit_start_topology_embeddings,
            first_hit_start_topology_pad_mask=first_hit_start_topology_pad_mask,
            first_hit_start_tree_graph_context=first_hit_start_tree_graph_context,
            phyla_global_context=phyla_global_context,
            phyla_clade_context=phyla_clade_context,
        )


def return_model(config):
    model_kwargs = dict(
        num_node_types=config["model"]["num_node_types"],
        num_edge_types=config["model"]["num_edge_types"],
        embed_dim=config["model"]["embed_dim"],
        output_dim=config["model"]["output_dim"],
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        dropout=config["model"].get("dropout", 0.1),
        attention_dropout=config["model"].get("attention_dropout", 0.1),
        activation_dropout=config["model"].get("activation_dropout", 0.1),
        drop_path_rate=config["model"].get("drop_path_rate", 0.1),
        use_performer=config["model"].get("use_performer", True),
        performer_nb_features=config["model"].get("performer_nb_features", 64),
        performer_generalized_attention=config["model"].get(
            "performer_generalized_attention", True
        ),
        layernorm_style=config["model"].get("layernorm_style", "prenorm"),
        tokenizer_lap_dim=config["model"]["tokenizer_lap_dim"],
        tokenizer_lap_dropout=config["model"]["tokenizer_lap_dropout"],
        tokenizer_n_layers=config["model"]["tokenizer_n_layers"],
        tokenizer_branch_length_mode=config["model"].get(
            "tokenizer_branch_length_mode", "linear"
        ),
        tokenizer_branch_length_num_buckets=config["model"].get(
            "tokenizer_branch_length_num_buckets", 64
        ),
        tokenizer_branch_length_log_min=config["model"].get(
            "tokenizer_branch_length_log_min", -8.0
        ),
        tokenizer_branch_length_log_max=config["model"].get(
            "tokenizer_branch_length_log_max", 1.0
        ),
        phyla_dim=config["model"]["phyla_dim"],
        phyla_use_leaf_tokens=config["model"].get("phyla_use_leaf_tokens", True),
        phyla_use_split_tokens=config["model"].get("phyla_use_split_tokens", True),
        phyla_leaf_scale=config["model"].get("phyla_leaf_scale", 1.0),
        phyla_split_scale=config["model"].get("phyla_split_scale", 1.0),
        phyla_use_global_context=config["model"].get(
            "phyla_use_global_context", False
        ),
        phyla_global_context_scale=config["model"].get(
            "phyla_global_context_scale", 1.0
        ),
        phyla_use_clade_context=config["model"].get(
            "phyla_use_clade_context", False
        ),
        phyla_clade_context_scale=config["model"].get(
            "phyla_clade_context_scale", 1.0
        ),
        autoregressive_head_mode=config["model"].get(
            "autoregressive_head_mode", "pairwise_threshold"
        ),
        autoregressive_group_refinement_layers=config["model"].get(
            "autoregressive_group_refinement_layers", 0
        ),
        autoregressive_max_subset_size=config["model"].get(
            "autoregressive_max_subset_size", 64
        ),
        autoregressive_use_case_conditioning=config["model"].get(
            "autoregressive_use_case_conditioning", False
        ),
        autoregressive_num_cases=config["model"].get("autoregressive_num_cases"),
        autoregressive_case_dim=config["model"].get("autoregressive_case_dim", 16),
        autoregressive_use_start_topology_conditioning=config["model"].get(
            "autoregressive_use_start_topology_conditioning", False
        ),
        autoregressive_start_topology_hidden_dim=config["model"].get(
            "autoregressive_start_topology_hidden_dim"
        ),
        autoregressive_start_topology_conditioning_mode=config["model"].get(
            "autoregressive_start_topology_conditioning_mode", "additive"
        ),
        autoregressive_start_topology_code_dim=config["model"].get(
            "autoregressive_start_topology_code_dim", 64
        ),
        autoregressive_frozen_start_case_embedding_path=config["model"].get(
            "autoregressive_frozen_start_case_embedding_path"
        ),
        autoregressive_frozen_start_case_adapter_mode=config["model"].get(
            "autoregressive_frozen_start_case_adapter_mode", "linear"
        ),
        autoregressive_frozen_start_case_adapter_hidden_dim=config["model"].get(
            "autoregressive_frozen_start_case_adapter_hidden_dim"
        ),
        first_hit_head_use_phase_input=config["model"].get(
            "first_hit_head_use_phase_input", False
        ),
        first_hit_head_phase_hidden_dim=config["model"].get(
            "first_hit_head_phase_hidden_dim"
        ),
        first_hit_head_mode=config["model"].get("first_hit_head_mode", "base"),
        first_hit_head_hidden_dim=config["model"].get("first_hit_head_hidden_dim"),
        first_hit_head_enable_refinement=config["model"].get(
            "first_hit_head_enable_refinement", False
        ),
        first_hit_head_refinement_layers=config["model"].get("first_hit_head_refinement_layers", 1),
        first_hit_head_router_hidden_dim=config["model"].get(
            "first_hit_head_router_hidden_dim"
        ),
        first_hit_head_num_cases=config["model"].get("first_hit_head_num_cases"),
        first_hit_head_case_dim=config["model"].get("first_hit_head_case_dim", 16),
        first_hit_frozen_start_case_embedding_path=config["model"].get(
            "first_hit_frozen_start_case_embedding_path"
        ),
        first_hit_frozen_start_case_adapter_mode=config["model"].get(
            "first_hit_frozen_start_case_adapter_mode", "linear"
        ),
        first_hit_frozen_start_case_adapter_hidden_dim=config["model"].get(
            "first_hit_frozen_start_case_adapter_hidden_dim"
        ),
        first_hit_start_tree_graph_detach=config["model"].get(
            "first_hit_start_tree_graph_detach", False
        ),
    )
    model_variant = str(config["model"].get("model_variant", "base"))
    if model_variant == "two_block_refine":
        model = TreeDenoiserTokenGTTwoBlock(
            **model_kwargs,
            block2_n_layers=config["model"].get("block2_n_layers"),
            block2_input_mode=config["model"].get("block2_input_mode", "direct"),
            block2_drop_path_rate=config["model"].get("block2_drop_path_rate"),
        )
    elif model_variant == "multi_block_refine":
        model = TreeDenoiserTokenGTMultiBlock(
            **model_kwargs,
            num_model_blocks=config["model"].get("num_model_blocks", 3),
            refine_block_n_layers=config["model"].get("refine_block_n_layers"),
            refine_block_input_mode=config["model"].get(
                "refine_block_input_mode", "direct"
            ),
            refine_block_drop_path_rate=config["model"].get(
                "refine_block_drop_path_rate"
            ),
        )
    else:
        model = TreeDenoiserTokenGT(**model_kwargs)

    return model
