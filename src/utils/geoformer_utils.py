import numpy as np
import torch

def nan_clean(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(a.astype(np.float32))
    a[np.abs(a) > 999] = 0
    return a

def loss_var(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    rmse = torch.mean((y_pred - y_true) ** 2, dim=[3, 4])
    rmse = rmse.sqrt().mean(dim=0)
    rmse = torch.sum(rmse, dim=[0, 1])
    return rmse

def loss_nino(y_pred_1d: torch.Tensor, y_true_1d: torch.Tensor) -> torch.Tensor:
    rmse = torch.sqrt(torch.mean((y_pred_1d - y_true_1d) ** 2, dim=0))
    return rmse.sum()

def build_nino_weight(T_out: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    base = np.array([1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6)
    w = base * np.log(np.arange(24) + 1)
    return torch.tensor(w[:T_out], device=device, dtype=dtype)

def calscore(nino_pred: torch.Tensor, nino_true: torch.Tensor, ninoweight: torch.Tensor) -> float:
    pred = nino_pred - nino_pred.mean(dim=0, keepdim=True)
    true = nino_true - nino_true.mean(dim=0, keepdim=True)
    cor = (pred * true).sum(dim=0) / (torch.sqrt((pred ** 2).sum(dim=0) * (true ** 2).sum(dim=0)) + 1e-6)
    acc = (ninoweight * cor).sum()
    rmse = torch.mean((nino_pred - nino_true) ** 2, dim=0).sqrt().sum()
    return (2.0 / 3.0 * acc - rmse).item()
