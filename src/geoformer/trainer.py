import math, torch
from transformers import Trainer
from utils.geoformer_utils import loss_var, loss_nino, build_nino_weight, calscore

class GeoformerTrainer(Trainer):
    def __init__(self, *args, data_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_cfg = data_cfg
        self.sstlevel = 2 if data_cfg["needtauxy"] else 0
        self.tf_decay = 2.5e-4

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer | None = None):
        from torch.optim.lr_scheduler import LambdaLR
        if optimizer is None:
            optimizer = self.optimizer  # HF 在 4.56 会把创建好的 optimizer 传进来

        d_size = self.model.config.d_size
        warmup = float(self.data_cfg["warmup"])
        factor = math.sqrt(d_size * warmup) * 0.0015

        def lr_lambda(step: int):
            step = max(step, 1)
            return factor * (d_size ** (-0.5)) * min(step ** (-0.5), step * (warmup ** (-1.5)))

        self.lr_scheduler = LambdaLR(optimizer, lr_lambda)
        return self.lr_scheduler

    def _nino_ts(self, field: torch.Tensor) -> torch.Tensor:
        lat0, lat1 = self.data_cfg["nino_region"]["lat"]
        lon0, lon1 = self.data_cfg["nino_region"]["lon"]
        sst = field[:, :, self.sstlevel]
        return sst[:, :, lat0:lat1, lon0:lon1].mean(dim=[2, 3])

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
        **kwargs,
    ):
        predictor = inputs["predictor"]
        labels    = inputs["labels"]
        tf_ratio = max(1.0 - self.state.global_step * self.tf_decay, 0.0)

        outputs = model(
            predictor=predictor,
            predictand=labels,
            teacher_forcing_ratio=tf_ratio,
            autoregressive=True,
            return_dict=True,
        )
        pred = outputs.prediction

        l_var = loss_var(pred, labels)
        nino_pred, nino_true = self._nino_ts(pred), self._nino_ts(labels)
        l_nino = loss_nino(nino_pred, nino_true)
        loss = l_var + l_nino

        # 可选：记录训练日志
        self.log({
            "train/loss_var": l_var.detach().item(),
            "train/loss_nino": l_nino.detach().item(),
            "train/tf_ratio": tf_ratio,
        })

        if return_outputs:
            return loss, outputs
        return loss

    @torch.no_grad()
    # 👇 同样加上 **kwargs，避免将来 Trainer 传入新参数时报错
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool = False,
        ignore_keys=None,
        **kwargs,
    ):
        predictor = inputs["predictor"]
        labels    = inputs["labels"]
        outputs = model(
            predictor=predictor,
            predictand=None,
            autoregressive=True,
            return_dict=True,
        )
        pred = outputs.prediction
        l_var = loss_var(pred, labels)
        nino_pred, nino_true = self._nino_ts(pred), self._nino_ts(labels)
        l_nino = loss_nino(nino_pred, nino_true)
        loss = (l_var + l_nino).detach()

        if prediction_loss_only:
            return loss, None, None
        # 让 Trainer 后处理为 numpy；返回 tensor 就可以
        return loss, pred.detach().cpu(), labels.detach().cpu()

def build_compute_metrics(data_cfg):
    def _fn(eval_pred):
        import torch
        y_pred = torch.from_numpy(eval_pred.predictions)
        y_true = torch.from_numpy(eval_pred.label_ids)
        lv = loss_var(y_pred, y_true).item()
        sst_pred, sst_true = y_pred[:, :, 2], y_true[:, :, 2]
        lat0, lat1 = data_cfg["nino_region"]["lat"]
        lon0, lon1 = data_cfg["nino_region"]["lon"]
        nino_pred = sst_pred[:, :, lat0:lat1, lon0:lon1].mean(dim=[2, 3])
        nino_true = sst_true[:, :, lat0:lat1, lon0:lon1].mean(dim=[2, 3])
        ln = loss_nino(nino_pred, nino_true).item()
        ninow = build_nino_weight(y_pred.size(1), device=torch.device("cpu"), dtype=y_pred.dtype)
        sc = calscore(nino_pred, nino_true, ninow)
        return {"loss_var": lv, "loss_nino": ln, "loss_com": lv+ln, "score": sc}
    return _fn
