import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

try:
    from transformers import EarlyStoppingCallback, TrainingArguments
except ImportError as exc:  # pragma: no cover - 错误提示即足够
    raise RuntimeError("未安装 transformers，请先执行 pip install transformers") from exc

from geoformer.modules import GeoformerConfig, GeoformerModel  # noqa: E402
from geoformer.dataset import (  # noqa: E402
    GeoformerCollator,
    LazyNetCDFEvalDataset,
    LazyNetCDFTrainDataset,
)
from geoformer.trainer import GeoformerTrainer, build_compute_metrics  # noqa: E402

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Geoformer training")
    parser.add_argument("--config-dir", type=Path, default=None, help="Directory containing data/model config JSON files")
    parser.add_argument("--model-config", type=str, default="model_config.json", help="Model config filename inside config dir")
    parser.add_argument("--data-config", type=str, default="data_config.json", help="Data config filename inside config dir")
    parser.add_argument("--num-train-epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Force Trainer to stop after N update steps")
    parser.add_argument("--limit-train-pairs", type=int, default=None, help="Limit training pairs sampled from dataset")
    parser.add_argument("--limit-eval-samples", type=int, default=None, help="Limit evaluation dataset length for smoke tests")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override Trainer output directory")
    parser.add_argument("--per-device-train-batch-size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None, help="Override eval batch size")
    parser.add_argument("--eval-steps", type=int, default=None, help="Override eval/save steps")
    parser.add_argument("--disable-wandb", action="store_true", help="Do not report metrics to Weights & Biases even if installed")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic seed override")
    parser.add_argument("--no-train", action="store_true", help="Initialise Trainer but skip the actual training loop")
    return parser.parse_args()


def _load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def _resolve_output_dir(data_cfg: dict, override: Optional[Path] = None) -> str:
    if override is not None:
        return str((PROJECT_ROOT / override).resolve() if not override.is_absolute() else override.resolve())

    raw_dir = data_cfg.get("save_dir")
    default_dir = PROJECT_ROOT / "checkpoints" / "geoformer_pretrain"
    if not raw_dir:
        return str(default_dir.resolve())

    candidate = Path(raw_dir)
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()

    # 如果配置仍指向旧的 ./model/，自动重定向到 checkpoints
    legacy_dir = (PROJECT_ROOT / "model").resolve()
    if candidate == legacy_dir:
        print("[INFO] Redirecting legacy save_dir './model/' to 'checkpoints/geoformer_pretrain'.")
        return str(default_dir.resolve())

    return str(candidate)


def _maybe_subset_eval(eval_dataset, max_items: int):
    if max_items is None:
        return eval_dataset
    if max_items <= 0:
        raise ValueError("limit-eval-samples should be > 0")
    from torch.utils.data import Subset

    max_index = min(len(eval_dataset), max_items)
    return Subset(eval_dataset, range(max_index))


def _resolve_report_to(disable_wandb: bool):
    if disable_wandb:
        return []
    try:
        import wandb  # noqa: F401
    except ImportError:
        print("[INFO] wandb not installed; metrics reporting disabled.")
        return []
    return ["wandb"]


def main():
    args = _parse_args()

    config_dir = args.config_dir or (PROJECT_ROOT / "configs")
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    model_cfg_json = _load_json(Path(config_dir) / args.model_config)
    data_cfg = _load_json(Path(config_dir) / args.data_config)

    if args.seed is not None:
        data_cfg["seed"] = args.seed

    if args.limit_train_pairs is not None:
        data_cfg["max_train_pairs"] = args.limit_train_pairs

    if args.per_device_train_batch_size is not None:
        data_cfg["batch_size_train"] = args.per_device_train_batch_size

    if args.per_device_eval_batch_size is not None:
        data_cfg["batch_size_eval"] = args.per_device_eval_batch_size

    if args.eval_steps is not None:
        data_cfg["eval_steps"] = args.eval_steps

    lat_len = data_cfg["lat_range"][1] - data_cfg["lat_range"][0]
    lon_len = data_cfg["lon_range"][1] - data_cfg["lon_range"][0]
    ps0, ps1 = model_cfg_json["patch_size"]
    model_cfg_json["emb_spatial_size"] = (lat_len // ps0) * (lon_len // ps1)

    model_cfg_json["input_length"] = data_cfg["input_length"]
    model_cfg_json["output_length"] = data_cfg["output_length"]
    model_cfg_json["needtauxy"] = data_cfg["needtauxy"]

    cfg = GeoformerConfig(**model_cfg_json)
    model = GeoformerModel(cfg)

    train_ds = LazyNetCDFTrainDataset(
        data_cfg["train_path"],
        data_cfg["lev_range"],
        data_cfg["lat_range"],
        data_cfg["lon_range"],
        data_cfg["input_length"],
        data_cfg["output_length"],
        data_cfg["needtauxy"],
        data_cfg.get("max_train_pairs", -1),
    )
    eval_ds = LazyNetCDFEvalDataset(
        data_cfg["eval_path"],
        data_cfg["lev_range"],
        data_cfg["lat_range"],
        data_cfg["lon_range"],
        data_cfg["input_length"],
        data_cfg["output_length"],
        data_cfg["needtauxy"],
    )

    eval_ds = _maybe_subset_eval(eval_ds, args.limit_eval_samples)

    collator = GeoformerCollator()
    training_args = TrainingArguments(
        output_dir=_resolve_output_dir(data_cfg, override=args.output_dir),
        per_device_train_batch_size=data_cfg["batch_size_train"],
        per_device_eval_batch_size=data_cfg["batch_size_eval"],
        num_train_epochs=args.num_train_epochs or data_cfg["num_epochs"],
        max_steps=args.max_steps if args.max_steps is not None else -1,
        eval_strategy="steps",
        eval_steps=data_cfg["eval_steps"],
        logging_steps=max(1, min(50, data_cfg["eval_steps"])),
        save_strategy="steps",
        save_steps=data_cfg["eval_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="score",
        greater_is_better=True,
        remove_unused_columns=False,
        learning_rate=1e-3,
        report_to=_resolve_report_to(args.disable_wandb),
        ddp_find_unused_parameters=False,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        seed=data_cfg.get("seed"),
    )

    trainer = GeoformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=build_compute_metrics(data_cfg),
        data_cfg=data_cfg,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=data_cfg.get("patience", 4))],
    )
    print("Trainer will use device:", trainer.args.device)
    if not args.no_train:
        trainer.train()
        print(trainer.evaluate())
        trainer.save_model(training_args.output_dir)
        cfg.save_pretrained(training_args.output_dir)
    else:
        print("[INFO] --no-train supplied; skipping training loop.")


if __name__ == "__main__":
    main()
