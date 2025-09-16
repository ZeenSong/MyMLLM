# History.md

## 概览

这份文件记录了与协作 AI 的关键对话结论、仓库整顿与已完成动作，便于后续追溯与交接。

## 目标与定位（已对齐）

* 目标：打造**海洋气象多模态大模型**，以 Geoformer 为视觉骨干，Qwen 为语言基座。
* 里程碑：视觉骨干稳定化 → Qwen 适配 → SFT 流水线 → 推理与部署。
* 流程：严格按“计划→提交→执行→提交”的工程化迭代；**Git 仅追踪代码**。

## 仓库梳理（达成的共识）

* `3D_Geoformer_SST_SSH-4090/`：无用但**保留**，已建议归档到 `archive/`。
* `Geoformer/`, `Mini-LLaVA/`, `Qwen2.5-VL/`：作为参考开源，放入 `third_party/`。
* `model/`：为检查点目录，统一迁入 `checkpoints/` 并从 Git 忽略。
* 活跃代码：`Geoformer_*` 系列与 `data_config.json`, `model_config.json`。
* 主目录下零散 `.py`：暂时无用，可移 `archive/scripts_unused/`。

## 实际操作与事件线

1. **最初环境问题**

   * 运行 `tree -L 2` 时出现 sandbox/landlock 报错（只读/受限环境）。
   * 发现 **Git 根目录指向上层**（`/data/users/.git`），导致在项目内 `git status` 出现 `../../XXX` 未跟踪文件。
   * 采取修复：在项目根重新初始化或调整仓库根，现已**修复为项目内 Git 根**。

2. **目录与文档初步搭建（规划与建议已形成）**

   * 规划将 `Geoformer_*.py` 归并至 `src/geoformer/`，工具脚本放入 `tools/`，`utils/` 独立。
   * 将 `data_config.json`, `model_config.json` 迁至 `configs/`；`model/` → `checkpoints/`。
   * `Geoformer/`, `Mini-LLaVA/`, `Qwen2.5-VL/` → `third_party/`；`3D_Geoformer_SST_SSH-4090/` → `archive/`。
   * 在 `docs/` 中添加了 `ROADMAP.md` 的骨架；创建了 `tools/train_geoformer.py` 的**脚手架**（用于从 `configs/` 读取配置并跑通最小训练流程，脚手架仍需在可写环境完善与验证）。

3. **开发规范与 Git 约束（已确认）**

   * 不包化（不生成 `pyproject.toml`），以 `PYTHONPATH=src` 运行。
   * `.gitignore` 仅追踪代码；忽略 `configs/*.json`（可保模板）、`checkpoints/`, `data/`, `third_party/`, `archive/`, `wandb/`。
   * 每一步骤前**先写计划到 `docs/`**，确认上一次已提交后再执行。

### 2025-09-16

* 新增 `tools/train_geoformer.py` 的命令行参数，支持覆写 epoch/steps、子样本数量与禁用 W&B，上线冒烟模式。
* 针对旧配置默认写入 `./model/` 的情况，训练脚本会自动重定向输出到 `checkpoints/geoformer_pretrain`，并允许通过 `--output-dir` 明确指定目录。
* 使用命令 `python tools/train_geoformer.py --max-steps 1 --limit-train-pairs 4 --limit-eval-samples 2 --eval-steps 1 --disable-wandb --per-device-train-batch-size 1 --per-device-eval-batch-size 1 --num-train-epochs 1 --output-dir checkpoints/geoformer_pretrain_smoke` 成功跑通 1 step 训练 + 2 step 评估，确认 Trainer 链路正常。
* 新增 `tools/run_geoformer_smoke.sh`，封装上述冒烟流程，可直接执行快速验证；训练与评估 loss 正常打印，长跑时去掉限制参数即可。
* 规划并搭建 Geoformer→SFT 自动标注链路：`src/data/sft_sampler.py` 提供随机采样与统计封装，`tools/build_sft_dataset.py` 输出具备多任务（预测+诊断）对话的 JSONL。
* 自动标注流程：历史窗口随机抽取可变长度片段，计算温度/风应力/Nino 指数统计，并随机生成“预测未来趋势”或“回顾历史特征”单任务中文问答；输出为 ShareGPT 样式 JSON，用户消息包含 `<geoformer>` 占位符，配套保存 `.npz` 特征文件（含历史+未来张量）以供 Geoformer 编码；示例命令 `python tools/build_sft_dataset.py --num-samples 32 --seed 42` 会在 `data/sft_raw/` 产出 `geoformer_sft.json` 及 `features/` 目录。

## 下一步建议（待执行）

* 在 `docs/` 新建当天的 `PLAN_YYYYMMDD.md`，列出**精确的文件移动与导入路径调整清单**。
* 完成目录移动与导入修正后，**跑一次最小冒烟**（dummy tensor）并记录到 `History.md`。
* 在 `docs/` 中补充“Geoformer→Qwen 适配契约”（张量形状、mask 规则、时间—空间维度）。
* 为 SFT 管线预留 `src/data/` 与 `configs/xxx.template.json` 模板。

## 里程碑状态（滚动更新）

* [x] 明确总体目标与阶段规划
* [x] 确立工程化协作与 Git 追踪边界
* [x] 定义目标目录结构与归档策略
* [ ] 完成文件物理迁移与导入路径统一
* [ ] 最小单测与冒烟
* [ ] 适配层雏形与契约文档
* [ ] SFT 数据 Schema 与训练脚本
* [ ] 推理脚本与示例
