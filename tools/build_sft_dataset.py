import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.sft_sampler import GeoformerSFTSampler, SFTSamplerConfig  # noqa: E402


def _default_dataset_path() -> Path:
    return PROJECT_ROOT / "dataset" / "3DGeoformer_origin_data" / "CMIP6_separate_model_up150m_tauxy_Nor_kb.nc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build automatically annotated SFT samples from Geoformer data")
    parser.add_argument("--dataset", type=Path, default=_default_dataset_path(), help="Path to the NetCDF dataset")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "sft_raw" / "geoformer_sft.json", help="Output dataset file (JSON/JSONL)")
    parser.add_argument("--feature-dir", type=Path, default=PROJECT_ROOT / "data" / "sft_raw" / "features", help="Directory to store Geoformer feature tensors")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of samples to generate")
    parser.add_argument("--input-length", type=int, default=None, help="固定历史窗口长度；与 --input-min/--input-max 互斥")
    parser.add_argument("--input-min", type=int, default=12, help="历史窗口最小长度")
    parser.add_argument("--input-max", type=int, default=18, help="历史窗口最大长度")
    parser.add_argument("--output-length", type=int, default=None, help="固定未来窗口长度；与 --output-min/--output-max 互斥")
    parser.add_argument("--output-min", type=int, default=16, help="未来窗口最小长度")
    parser.add_argument("--output-max", type=int, default=24, help="未来窗口最大长度")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output if args.output.is_absolute() else (PROJECT_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_dir = args.feature_dir if args.feature_dir.is_absolute() else (PROJECT_ROOT / args.feature_dir).resolve()
    feature_dir.mkdir(parents=True, exist_ok=True)

    if args.input_length is not None:
        args.input_min = args.input_max = args.input_length
    if args.output_length is not None:
        args.output_min = args.output_max = args.output_length

    if args.input_min > args.input_max:
        raise ValueError("input-min should be <= input-max")
    if args.output_min > args.output_max:
        raise ValueError("output-min should be <= output-max")

    config = SFTSamplerConfig(
        dataset_path=args.dataset if args.dataset.is_absolute() else (PROJECT_ROOT / args.dataset).resolve(),
        min_input_length=args.input_min,
        max_input_length=args.input_max,
        min_output_length=args.output_min,
        max_output_length=args.output_max,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    sampler = GeoformerSFTSampler(config)

    records = []
    for item in sampler.sample():
        feature_path = feature_dir / f"{item.sample_id}.npz"
        np.savez_compressed(
            feature_path,
            history=item.history,
            future=item.future,
        )

        try:
            rel_feature = feature_path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_feature = feature_path
        record = {
            "messages": [
                {"role": "user", "content": item.prompt},
                {"role": "assistant", "content": item.response},
            ],
            "geoformer": [str(rel_feature)],
            "task": item.task,
            "metadata": item.metadata,
        }
        records.append(record)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(records)} samples to {output_path}")


if __name__ == "__main__":
    main()
