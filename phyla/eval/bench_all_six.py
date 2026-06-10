#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from phyla.utils.utils import load_config
from phyla.eval.evo_reasoning_eval import (
    Config,
    load_model,
    tree_reconstruction_benchmark,
)


def load_list(p: Path) -> list[str]:
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]


def run_one(
    *,
    cfg_path: str,
    ckpt: str | None,
    dataset_type: str,
    dataset_names: list[str],
    force_encoding: str,
    label: str,
    device: str | None,
):
    orig = sys.argv
    sys.argv = ["x", cfg_path]
    config = load_config(Config)
    sys.argv = orig

    if device:
        config.eval.device = device

    config.trainer.checkpoint_path = ckpt
    models = {"Phyla": load_model(config=config, random_model=(ckpt is None))}

    # TreeFam needs dictionary_data; evo_reasoning_eval will download treefam.pickle if missing.
    dictionary_data = None

    normrfs, _, _ = tree_reconstruction_benchmark(
        models,
        num_datasets=[0, len(dataset_names)],
        # output_file_name=f"benchmark_results_{label}_{dataset_type}.tsv",
        output_file_name=None,
        dataset_type=dataset_type,
        dictionary_data=dictionary_data,
        device=config.eval.device,
        dataset_names=dataset_names,
        convert_to_aa=False,
        force_encoding=force_encoding,
    )
    avg = sum(normrfs) / len(normrfs)
    print(
        f"{label}: avg_normrf={avg:.6f} successes={len(normrfs)}/{len(dataset_names)}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sample_eval_config.yaml")
    ap.add_argument(
        "--ckpt_4k",
        required=True,
        help="Path to 4k checkpoint .ckpt file (download from HF).",
    )
    ap.add_argument("--device", default=None, help="Override eval device, e.g. cuda:0")
    args = ap.parse_args()

    tb_nuc = load_list(
        Path("phyla/eval/eval_preds/treebase/treebase_nucleotide_datasets.txt")
    )
    tb_prot = load_list(
        Path("phyla/eval/eval_preds/treebase/treebase_protein_datasets.txt")
    )
    tf_prot = load_list(
        Path("phyla/eval/eval_preds/treefam/treefam_protein_datasets.txt")
    )

    # Random
    # run_one(
    #     cfg_path=args.config,
    #     ckpt=None,
    #     dataset_type="treebase",
    #     dataset_names=tb_prot,
    #     force_encoding="protein",
    #     label="TREEBASE_PROTEIN_RANDOM",
    #     device=args.device,
    # )
    # run_one(
    #     cfg_path=args.config,
    #     ckpt=None,
    #     dataset_type="treebase",
    #     dataset_names=tb_nuc,
    #     force_encoding="nucleotide",
    #     label="TREEBASE_TRUE_NUC_RANDOM",
    #     device=args.device,
    # )
    # run_one(
    #     cfg_path=args.config,
    #     ckpt=None,
    #     dataset_type="treefam",
    #     dataset_names=tf_prot,
    #     force_encoding="protein",
    #     label="TREEFAM_PROTEIN_RANDOM",
    #     device=args.device,
    # )

    # 4k checkpoint
    run_one(
        cfg_path=args.config,
        ckpt=args.ckpt_4k,
        dataset_type="treebase",
        dataset_names=tb_prot,
        force_encoding="protein",
        label="TREEBASE_PROTEIN_4K",
        device=args.device,
    )
    run_one(
        cfg_path=args.config,
        ckpt=args.ckpt_4k,
        dataset_type="treebase",
        dataset_names=tb_nuc,
        force_encoding="nucleotide",
        label="TREEBASE_TRUE_NUC_4K",
        device=args.device,
    )
    # run_one(
    #     cfg_path=args.config,
    #     ckpt=args.ckpt_4k,
    #     dataset_type="treefam",
    #     dataset_names=tf_prot,
    #     force_encoding="protein",
    #     label="TREEFAM_PROTEIN_4K",
    #     device=args.device,
    # )


if __name__ == "__main__":
    main()
