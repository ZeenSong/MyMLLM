import json
import os
import sys
from pathlib import Path

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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def _load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def _resolve_output_dir(data_cfg: dict) -> str:
    raw_dir = data_cfg.get("save_dir")
    if raw_dir:
        candidate = Path(raw_dir)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return str(candidate.resolve())
    return str((PROJECT_ROOT / "checkpoints" / "geoformer_pretrain").resolve())


def main():
    config_dir = PROJECT_ROOT / "configs"
    model_cfg_json = _load_json(config_dir / "model_config.json")
    data_cfg = _load_json(config_dir / "data_config.json")

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

    collator = GeoformerCollator()
    args = TrainingArguments(
        output_dir=_resolve_output_dir(data_cfg),
        per_device_train_batch_size=data_cfg["batch_size_train"],
        per_device_eval_batch_size=data_cfg["batch_size_eval"],
        num_train_epochs=data_cfg["num_epochs"],
        eval_strategy="steps",
        eval_steps=data_cfg["eval_steps"],
        logging_steps=50,
        save_strategy="steps",
        save_steps=data_cfg["eval_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="score",
        greater_is_better=True,
        remove_unused_columns=False,
        learning_rate=1e-3,
        report_to=["wandb"],
        ddp_find_unused_parameters=False,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
    )

    trainer = GeoformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=build_compute_metrics(data_cfg),
        data_cfg=data_cfg,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=data_cfg.get("patience", 4))],
    )
    print("Trainer will use device:", trainer.args.device)
    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(args.output_dir)
    cfg.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
