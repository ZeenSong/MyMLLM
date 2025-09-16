# AgentProtocol.md

## 目标与背景

* **总目标**：基于 **Geoformer** 视觉编码与 **Qwen** 系列语言基座，打造一个面向**海洋气象**的多模态大模型（预训练→SFT→推理）。
* **当前状态**：已将 Geoformer 预训练流程**移植到 🤗 Transformers**；正在聚焦 Geoformer 视觉骨干与 Qwen 的对接，以及后续 SFT 管线。
* **协作原则**：严格遵循“先规划、后提交计划、确认已提交、再执行、执行后提交”的软件工程流程；**不随意改动第三方代码**；**Git 仅追踪代码**。全程使用中文对话。

## 仓库结构（约定）

* `src/`：自研源码

  * `geoformer/`：`Geoformer_*.py`（modules/dataset/trainer 等，已归并为活动路径）
  * `qwen_integration/`：后续的 Geoformer→Qwen 适配层
  * `utils/`：`geoformer_utils.py` 等工具
* `tools/`：可执行脚本（如 `train_geoformer.py`）
* `configs/`：`data_config.json`, `model_config.json`（**默认不纳入 Git**，可保留模板）
* `checkpoints/`：训练产物与权重（**不纳入 Git**）
* `third_party/`：只读参考代码（`Geoformer/`, `Mini-LLaVA/`, `Qwen2.5-VL/`）
* `archive/3D_Geoformer_SST_SSH-4090/`：无用但需保留的旧资料
* `scripts/`：仅保留当下必需脚本，其余移入 `archive/scripts_unused/`
* `docs/`：文档（`ROADMAP.md`, `AgentProtocol.md`, `History.md` 等）

> 备注：当前活跃文件为 `Geoformer_*.py` 与 `data_config.json`, `model_config.json`；主目录下其他散落的 `.py` 暂时无用。

## 工作流与提交规范

1. **计划先行**：将“将要执行的动作与预期改动范围”写入 `docs/` 下的新 Markdown（如 `PLAN_YYYYMMDD.md`）。
2. **确认干净**：确保当前 Git 工作区**已提交**且干净。
3. **再执行**：按计划执行，**仅修改自研代码路径**；第三方与检查点不改动。
4. **复核与提交**：完成后核对改动，撰写精炼 commit：

   * `chore: scaffold repository directories`
   * `refactor: move geoformer modules into src`
   * `docs: add roadmap and repo layout`
5. **分支策略**：以功能分支推进（如 `feat/repo-structure`, `feat/qwen-vision-adapter`），`main` 保持可用。
6. **只追踪代码**：通过 `.gitignore` 忽略 `configs/*.json`（除模板）、`checkpoints/`, `data/`, `third_party/`, `archive/`, `wandb/`。

## 编码与命名（简要）

* **风格**：Python 采用 PEP8；类型注解尽量完整；函数/类/模块注释到位。
* **命名**：模块与目录用小写下划线；类名用驼峰；常量全大写。
* **日志与监控**：训练脚本支持 `W&B` 或最少级别的 `logging` 打点。
* **可复现性**：在工具脚本中提供最小可运行示例（形状/步数/路径），并兼容 `PYTHONPATH=src` 运行。

## 接口与测试（最低保障）

* **前向冒烟测试**：`GeoformerModel.forward` 支持 dummy 输入张量的最小单测，校验张量尺寸与 mask。
* **适配层契约**：Geoformer→Qwen 的 embedding、padding/mask、时空维度对齐需在文档中显式说明。

## 你应该/不应该做什么

* ✅ 先写计划 → 确认已提交 → 执行 → 复核 → 提交
* ✅ 对每次“目录移动/接口变更”写明**动机、范围、回滚方法**
* ❌ 不直接修改 `third_party/`
* ❌ 不提交大文件、数据、检查点、真实配置
* ❌ 不在未提交上一个步骤成果前，开启新的大改动