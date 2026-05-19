import os
import sys
import logging
import torch

from ete3 import Tree as EteTree
from utils.random_tree import Tree


logger = logging.getLogger(__name__)
_LOCAL_MAMBA_RUNTIME_PATH = "/tmp/mamba_src/mamba_ssm-2.3.1"


def _ensure_local_mamba_runtime_path():
    if not os.path.isdir(_LOCAL_MAMBA_RUNTIME_PATH):
        return
    while _LOCAL_MAMBA_RUNTIME_PATH in sys.path:
        sys.path.remove(_LOCAL_MAMBA_RUNTIME_PATH)
    sys.path.insert(0, _LOCAL_MAMBA_RUNTIME_PATH)


def _encode_sequences_phyla_direct(sequences, names):
    from phyla.model.model import Phyla

    encoded_aa = []
    cls_token_mask = []
    sequence_mask = []
    for idx, seq in enumerate(sequences):
        encoded_aa.append(22)
        cls_token_mask.append(1)
        sequence_mask.append(idx)
        for aa in str(seq or "").upper():
            encoded_aa.append(Phyla.amino_acid_encoding.get(aa, 23))
            cls_token_mask.append(0)
            sequence_mask.append(idx)

    batch = {
        "encoded_sequences": torch.IntTensor(encoded_aa).unsqueeze(0),
        "cls_positions": torch.IntTensor(cls_token_mask).unsqueeze(0),
        "sequence_mask": torch.IntTensor(sequence_mask).unsqueeze(0),
    }
    return batch, names


def _normalize_phyla_state_dict_for_model(raw_state_dict, model):
    target_keys = set(model.state_dict().keys())
    candidates = [
        ("identity", ()),
        ("strip_model", ("model.",)),
        ("strip_forward_module_model", ("_forward_module.model.",)),
        ("strip_forward_module", ("_forward_module.",)),
        ("strip_module_model", ("module.model.",)),
        ("strip_module", ("module.",)),
    ]
    best_name = None
    best_state_dict = None
    best_score = None
    for name, prefixes in candidates:
        normalized = {}
        for key, value in raw_state_dict.items():
            normalized_key = str(key)
            for prefix in prefixes:
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix) :]
            normalized[normalized_key] = value
        normalized_keys = set(normalized)
        missing = target_keys - normalized_keys
        unexpected = normalized_keys - target_keys
        score = (len(target_keys & normalized_keys), -len(missing), -len(unexpected))
        if best_score is None or score > best_score:
            best_name = name
            best_state_dict = normalized
            best_score = score
    return best_name, best_state_dict


def _load_phyla_model_direct(config, random_model=False):
    from phyla import phyla

    model_name = str(config.model.model_name)
    known_model = model_name.lower() in {"phyla-alpha", "phyla-beta"}
    if known_model:
        config.model.vocab_size = 24
    model = phyla(
        name=model_name,
        config=config,
        custom_arch=not known_model and bool(config.trainer.checkpoint_path),
        device=config.eval.device,
    )
    if not random_model and config.trainer.checkpoint_path:
        checkpoint = torch.load(config.trainer.checkpoint_path, map_location="cpu")
        raw_state_dict = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint
            else checkpoint
        )
        normalization, state_dict = _normalize_phyla_state_dict_for_model(
            raw_state_dict,
            model,
        )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            logger.warning(
                "Loaded Phyla checkpoint with normalization=%s, missing=%s, unexpected=%s",
                normalization,
                len(missing),
                len(unexpected),
            )
    model = model.to(config.eval.device)
    model.eval()
    return {"model": model, "alphabet_tokenizer": None}


def _load_phyla_runtime():
    _ensure_local_mamba_runtime_path()
    from phyla.utils.utils import load_config
    from phyla.utils.eval_configs import Config

    return load_config, Config, _load_phyla_model_direct, _encode_sequences_phyla_direct


class PhylaEmbeddingMixin:
    def compute_phyla_embeddings(self, sequences, names, device="cuda"):
        """
        Generates Phyla embeddings for a batch of sequences.
        """
        if self.phyla_model is None:
            raise ValueError("Phyla model not loaded.")

        # This utility handles tokenization, padding, and CLS token placement
        _, _, _, _encode_sequences_openfold_style = _load_phyla_runtime()
        batch, _ = _encode_sequences_openfold_style(sequences, names)

        # Generate Embeddings
        with torch.no_grad():
            encoded_seqs = batch["encoded_sequences"].to(device)
            sequence_mask = batch["sequence_mask"].to(device)
            cls_positions = batch["cls_positions"].bool().to(device)

            self.phyla_model.to(device)

            # Handle different forward pass signatures depending on model wrapper
            if "TrainingModule" in str(type(self.phyla_model)):
                embeddings = self.phyla_model(
                    encoded_seqs,
                    cls_token_mask=cls_positions,
                    sequence_mask=sequence_mask,
                )
            else:
                embeddings = self.phyla_model(
                    encoded_seqs,
                    sequence_mask,
                    cls_positions,
                )

        return embeddings

    def _load_precomputed_phyla_embeddings(self, path):
        if os.path.isdir(path):
            pt_paths = [
                os.path.join(path, name)
                for name in sorted(os.listdir(path))
                if name.endswith(".pt") and not name.startswith("._")
            ]
            if not pt_paths:
                raise ValueError(f"No .pt Phyla embedding files found in {path}.")
            by_dataset_id = {}
            global_by_name = {}
            conflicting_names = set()
            for pt_path in pt_paths:
                name_to_embedding, dataset_id = self._read_precomputed_phyla_embeddings(
                    pt_path
                )
                if dataset_id is None:
                    dataset_id = os.path.basename(pt_path).split("_", 1)[0]
                by_dataset_id[str(dataset_id).upper()] = name_to_embedding
                for name, embedding in name_to_embedding.items():
                    if name in conflicting_names:
                        continue
                    previous = global_by_name.get(name)
                    if previous is not None and not torch.allclose(previous, embedding):
                        global_by_name.pop(name, None)
                        conflicting_names.add(name)
                        continue
                    global_by_name[name] = embedding.clone()
            self.phyla_precomputed_by_dataset_id = by_dataset_id
            self.phyla_precomputed_name_to_embedding = global_by_name
            return

        name_to_embedding, dataset_id = self._read_precomputed_phyla_embeddings(path)
        self.phyla_precomputed_name_to_embedding = name_to_embedding
        if dataset_id is not None:
            self.phyla_precomputed_by_dataset_id = {
                str(dataset_id).upper(): name_to_embedding
            }

    def _read_precomputed_phyla_embeddings(self, path):
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                "Expected a dict payload with 'sequence_names' and 'embeddings'."
            )

        sequence_names = payload.get("sequence_names")
        if sequence_names is None:
            sequence_names = payload.get("names")
        embeddings = payload.get("embeddings")
        if embeddings is None:
            embeddings = payload.get("phyla_embeddings")
        if sequence_names is None or embeddings is None:
            raise ValueError(
                "Precomputed Phyla file must contain 'sequence_names' and 'embeddings'."
            )

        if torch.is_tensor(embeddings):
            tensor = embeddings.detach().cpu().float()
        else:
            tensor = torch.as_tensor(embeddings, dtype=torch.float32)

        if tensor.dim() == 3:
            if tensor.size(0) != 1:
                raise ValueError(
                    f"Expected embeddings with leading batch size 1, got {tuple(tensor.shape)}."
                )
            tensor = tensor.squeeze(0)
        if tensor.dim() != 2:
            raise ValueError(
                f"Expected embeddings with shape [N, D], got {tuple(tensor.shape)}."
            )
        if len(sequence_names) != tensor.size(0):
            raise ValueError(
                f"Sequence name count {len(sequence_names)} does not match "
                f"embedding rows {tensor.size(0)}."
            )

        expected_dim = int(self.model.phyla_proj.in_features)
        if tensor.size(1) != expected_dim:
            raise ValueError(
                f"Precomputed embedding dim {tensor.size(1)} does not match "
                f"model phyla_dim {expected_dim}."
            )

        name_to_embedding = {
            str(name): tensor[idx].clone()
            for idx, name in enumerate(sequence_names)
        }
        return name_to_embedding, payload.get("dataset_id")

    def _ordered_leaf_names_from_mapping(self, mapping, num_leaf=None):
        if mapping is None:
            return None
        ordered = []
        limit = None if num_leaf is None else int(num_leaf)
        for raw_idx, raw_name in mapping.items():
            if raw_name in (None, "", "ROOT_DUMMY"):
                continue
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                continue
            if limit is not None and idx >= limit:
                continue
            ordered.append((idx, str(raw_name)))
        if not ordered:
            return None
        ordered.sort(key=lambda item: item[0])
        return [name for _idx, name in ordered]

    def _ordered_leaf_names_from_newick(self, newick_tree):
        tree = Tree(newick_tree)
        names = []
        for idx in range(tree.n_leaves):
            name = str(tree.id_to_name[idx])
            if name != "ROOT_DUMMY":
                names.append(name)
        return names

    def _lookup_precomputed_phyla_embeddings(
        self,
        names,
        device=None,
        dataset_id=None,
    ):
        if not names:
            return None
        if dataset_id is not None:
            by_dataset_id = getattr(self, "phyla_precomputed_by_dataset_id", {}) or {}
            precomputed = by_dataset_id.get(str(dataset_id).upper())
            if precomputed is not None:
                missing = [
                    str(name)
                    for name in names
                    if str(name) not in precomputed
                ]
                if not missing:
                    embeddings = torch.stack(
                        [
                            precomputed[str(name)]
                            for name in names
                        ],
                        dim=0,
                    )
                    if device is not None:
                        embeddings = embeddings.to(device)
                    return embeddings

        precomputed = getattr(self, "phyla_precomputed_name_to_embedding", None)
        if not precomputed:
            return None
        missing = [
            str(name)
            for name in names
            if str(name) not in precomputed
        ]
        if missing:
            return None
        embeddings = torch.stack(
            [
                precomputed[str(name)]
                for name in names
            ],
            dim=0,
        )
        if device is not None:
            embeddings = embeddings.to(device)
        return embeddings

    def _resolve_precomputed_phyla_embeddings_for_tree(
        self,
        newick_tree,
        mapping=None,
        num_leaf=None,
        device=None,
        dataset_id=None,
    ):
        names = self._ordered_leaf_names_from_mapping(mapping, num_leaf=num_leaf)
        if names is None and newick_tree is not None:
            names = self._ordered_leaf_names_from_newick(newick_tree)
        if not names:
            return None
        embeddings = self._lookup_precomputed_phyla_embeddings(
            names,
            device=device,
            dataset_id=dataset_id,
        )
        if embeddings is None:
            return None
        return embeddings.unsqueeze(0)
