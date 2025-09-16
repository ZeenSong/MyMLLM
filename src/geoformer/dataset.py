# geoformer_dataset_clean.py
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Sequence
from dataclasses import dataclass

# ------------------ 工具 ------------------ #
def _nan_clean(a: np.ndarray) -> np.ndarray:
    # 统一清洗：NaN/Inf -> 0，并把异常大值裁掉（与旧代码一致）
    a = np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    a[np.abs(a) > 999] = 0
    return a

_TIME_CANDS   = {"time", "lead", "n_mon", "month", "step", "t"}
_MODEL_CANDS  = {"n_model", "model", "member", "ens", "ensemble", "n_yr", "year", "n_year", "sample"}
_SAMPLE_CANDS = {"n_sample", "sample", "case", "n_case", "n_member"}

def _infer_dims(var: xr.DataArray, want_has_lev: bool):
    """
    根据 DataArray 的 dims 自动识别：
      - 有 lev 的变量（temperature*）：dims 应含 lev/lat/lon + 2 个其它维（样本维、时间维）
      - 无 lev 的变量（taux*/tauy*）：dims 应含 lat/lon + 2 个其它维（样本维、时间维）
    返回: (sample_dim, time_dim, lev_dim, lat_dim, lon_dim)
    其中没有的（例如 taux/tauy 的 lev_dim）返回 None
    """
    dims = tuple(var.dims)
    lat_dim = "lat" if "lat" in dims else None
    lon_dim = "lon" if "lon" in dims else None
    lev_dim = "lev" if ("lev" in dims and want_has_lev) else (None if not want_has_lev else ("lev" if "lev" in dims else None))

    ignore = {d for d in [lev_dim, lat_dim, lon_dim] if d is not None}
    others = [d for d in dims if d not in ignore]

    # 识别时间/样本维
    time_dim_named   = [d for d in others if d in _TIME_CANDS]
    sample_dim_named = [d for d in others if d in (_MODEL_CANDS | _SAMPLE_CANDS)]
    time_dim   = time_dim_named[0]   if time_dim_named   else None
    sample_dim = sample_dim_named[0] if sample_dim_named else None

    remaining = [d for d in others if d not in {time_dim, sample_dim} if d is not None]
    sizes = var.sizes
    if time_dim is None and sample_dim is None and len(remaining) >= 2:
        a, b = remaining[:2]
        # 启发式：较长者当时间维
        time_dim   = a if sizes[a] >= sizes[b] else b
        sample_dim = b if time_dim == a else a
    elif time_dim is None and remaining:
        time_dim = remaining[0]
    elif sample_dim is None and remaining:
        sample_dim = remaining[0]

    return sample_dim, time_dim, lev_dim, lat_dim, lon_dim


# ------------------ 懒加载训练集 ------------------ #
class LazyNetCDFTrainDataset(Dataset):
    """
    训练集懒加载：
      - __init__ 只读元信息与 shape
      - __getitem__ 按需窗口读取
      - 自动识别变量维度名（兼容 n_model/n_mon 或 time/lead）
      - 全部通过 xarray.transpose(命名维顺序) 排序，杜绝 moveaxis/轴越界
    """
    def __init__(
        self,
        nc_path: str,
        lev_range: Tuple[int, int],
        lat_range: Tuple[int, int],
        lon_range: Tuple[int, int],
        input_length: int,
        output_length: int,
        needtauxy: bool,
        max_train_pairs: int = -1,
        engine: str = "netcdf4",
    ):
        self.path = nc_path
        self.engine = engine
        self.lr, self.latr, self.lonr = lev_range, lat_range, lon_range
        self.T_in, self.T_out = input_length, output_length
        self.needtauxy = needtauxy

        with xr.open_dataset(self.path, engine=self.engine) as ds:
            # 温度维度名
            temp = ds["temperatureNor"]
            sdim, tdim, self.temp_lev, self.temp_lat, self.temp_lon = _infer_dims(temp, want_has_lev=True)
            if sdim is None or tdim is None:
                raise ValueError(f"无法从 temperatureNor 的 dims {temp.dims} 推断样本/时间维")
            self.sample_dim, self.time_dim = sdim, tdim

            sizes = temp.sizes
            self.M = int(sizes[self.sample_dim])
            self.T = int(sizes[self.time_dim])

            # 风应力维度名（无 lev）
            if self.needtauxy:
                taux = ds["tauxNor"]; tauy = ds["tauyNor"]
                self.taux_sdim, self.taux_tdim, _, self.taux_lat, self.taux_lon = _infer_dims(taux, want_has_lev=False)
                self.tauy_sdim, self.tauy_tdim, _, self.tauy_lat, self.tauy_lon = _infer_dims(tauy, want_has_lev=False)

        # 生成 (m,t) 训练窗口索引
        st_min = self.T_in - 1
        ed_max = self.T - self.T_out
        if ed_max <= st_min:
            raise ValueError(f"可用时间长度不足：总 T={self.T}, 需要 T_in={self.T_in}, T_out={self.T_out}")
        pairs = [(m, t) for m in range(self.M) for t in range(st_min, ed_max)]
        if max_train_pairs and max_train_pairs > 0 and len(pairs) > max_train_pairs:
            import random
            pairs = random.sample(pairs, max_train_pairs)
        self.indices = np.asarray(pairs, dtype=np.int32)

        self._ds = None  # per-worker 懒打开

    def _ensure_open(self):
        if self._ds is None:
            self._ds = xr.open_dataset(self.path, engine=self.engine, cache=False)

    def __len__(self): return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        self._ensure_open()
        m, t = map(int, self.indices[i])

        # -------- X: temperatureNor -> (T_in, L, H, W)
        x_temp = (
            self._ds["temperatureNor"]
            .isel(
                **{
                    self.sample_dim: m,
                    self.time_dim: slice(t - self.T_in + 1, t + 1),
                    self.temp_lev: slice(self.lr[0],  self.lr[1]),
                    self.temp_lat: slice(self.latr[0], self.latr[1]),
                    self.temp_lon: slice(self.lonr[0], self.lonr[1]),
                }
            )
            .transpose(self.time_dim, self.temp_lev, self.temp_lat, self.temp_lon)  # 关键：按名字排轴
            .values
        )
        x_temp = _nan_clean(x_temp)

        if self.needtauxy:
            # -------- X: tauxNor/tauyNor -> (T_in, H, W)
            x_taux = (
                self._ds["tauxNor"]
                .isel(
                    **{
                        self.taux_sdim: m,
                        self.taux_tdim: slice(t - self.T_in + 1, t + 1),
                        self.taux_lat:  slice(self.latr[0], self.latr[1]),
                        self.taux_lon:  slice(self.lonr[0], self.lonr[1]),
                    }
                )
                .transpose(self.taux_tdim, self.taux_lat, self.taux_lon)
                .values
            )
            x_tauy = (
                self._ds["tauyNor"]
                .isel(
                    **{
                        self.tauy_sdim: m,
                        self.tauy_tdim: slice(t - self.T_in + 1, t + 1),
                        self.tauy_lat:  slice(self.latr[0], self.latr[1]),
                        self.tauy_lon:  slice(self.lonr[0], self.lonr[1]),
                    }
                )
                .transpose(self.tauy_tdim, self.tauy_lat, self.tauy_lon)
                .values
            )
            x_taux = _nan_clean(x_taux)
            x_tauy = _nan_clean(x_tauy)

            # 拼成 (T_in, C, H, W)
            x = np.concatenate([x_taux[:, None], x_tauy[:, None], x_temp], axis=1).astype(np.float32)
        else:
            x = x_temp.astype(np.float32)

        # -------- Y: temperatureNor -> (T_out, L, H, W)
        y_temp = (
            self._ds["temperatureNor"]
            .isel(
                **{
                    self.sample_dim: m,
                    self.time_dim: slice(t + 1, t + 1 + self.T_out),
                    self.temp_lev: slice(self.lr[0],  self.lr[1]),
                    self.temp_lat: slice(self.latr[0], self.latr[1]),
                    self.temp_lon: slice(self.lonr[0], self.lonr[1]),
                }
            )
            .transpose(self.time_dim, self.temp_lev, self.temp_lat, self.temp_lon)
            .values
        )
        y_temp = _nan_clean(y_temp)

        if self.needtauxy:
            y_taux = (
                self._ds["tauxNor"]
                .isel(
                    **{
                        self.taux_sdim: m,
                        self.taux_tdim: slice(t + 1, t + 1 + self.T_out),
                        self.taux_lat:  slice(self.latr[0], self.latr[1]),
                        self.taux_lon:  slice(self.lonr[0], self.lonr[1]),
                    }
                )
                .transpose(self.taux_tdim, self.taux_lat, self.taux_lon)
                .values
            )
            y_tauy = (
                self._ds["tauyNor"]
                .isel(
                    **{
                        self.tauy_sdim: m,
                        self.tauy_tdim: slice(t + 1, t + 1 + self.T_out),
                        self.tauy_lat:  slice(self.latr[0], self.latr[1]),
                        self.tauy_lon:  slice(self.lonr[0], self.lonr[1]),
                    }
                )
                .transpose(self.tauy_tdim, self.tauy_lat, self.tauy_lon)
                .values
            )
            y_taux = _nan_clean(y_taux)
            y_tauy = _nan_clean(y_tauy)

            y = np.concatenate([y_taux[:, None], y_tauy[:, None], y_temp], axis=1).astype(np.float32)
        else:
            y = y_temp.astype(np.float32)

        # 形状断言（调试一次即可）
        # x: (T_in, C, H, W)  y: (T_out, C, H, W)
        assert x.ndim == 4 and y.ndim == 4, f"Got x={x.shape}, y={y.shape}"
        return x, y

    def __del__(self):
        try:
            if self._ds is not None:
                self._ds.close()
        except Exception:
            pass


# ------------------ 懒加载评估集 ------------------ #
class LazyNetCDFEvalDataset(Dataset):
    """
    评估集懒加载：自动识别 *_in / *_out 维度名
      - 统一用 transpose(命名维) → .values
      - 统一清洗 NaN/Inf
    """
    def __init__(
        self,
        nc_path: str,
        lev_range: Tuple[int, int],
        lat_range: Tuple[int, int],
        lon_range: Tuple[int, int],
        input_length: int,
        output_length: int,
        needtauxy: bool,
        engine: str = "netcdf4",
    ):
        self.path = nc_path
        self.engine = engine
        self.lr, self.latr, self.lonr = lev_range, lat_range, lon_range
        self.T_in, self.T_out = input_length, output_length
        self.needtauxy = needtauxy

        with xr.open_dataset(self.path, engine=self.engine) as ds:
            tin  = ds["temperatureNor_in"]
            tout = ds["temperatureNor_out"]
            self.sdim_in,  self.tdim_in,  self.lev_in,  self.lat_in,  self.lon_in  = _infer_dims(tin,  want_has_lev=True)
            self.sdim_out, self.tdim_out, self.lev_out, self.lat_out, self.lon_out = _infer_dims(tout, want_has_lev=True)
            if self.sdim_in is None:
                others = [d for d in tin.dims  if d not in {self.tdim_in,  self.lev_in,  self.lat_in,  self.lon_in}]
                self.sdim_in = others[0]
            if self.sdim_out is None:
                others = [d for d in tout.dims if d not in {self.tdim_out, self.lev_out, self.lat_out, self.lon_out}]
                self.sdim_out = others[0]
            self.N = int(tin.sizes[self.sdim_in])

            if self.needtauxy:
                self.taux_in  = ds["tauxNor_in"]
                self.tauy_in  = ds["tauyNor_in"]
                self.taux_out = ds["tauxNor_out"]
                self.tauy_out = ds["tauyNor_out"]
                self.s_in,  self.t_in,  _, self.lat_in_tau,  self.lon_in_tau  = _infer_dims(self.taux_in,  want_has_lev=False)
                self.s_out, self.t_out, _, self.lat_out_tau, self.lon_out_tau = _infer_dims(self.taux_out, want_has_lev=False)

        self._ds = None

    def _ensure_open(self):
        if self._ds is None:
            self._ds = xr.open_dataset(self.path, engine=self.engine, cache=False)

    def __len__(self): return self.N

    def __getitem__(self, i: int):
        self._ensure_open()
        # X: temp_in -> (T_in, L, H, W)
        x_temp = (
            self._ds["temperatureNor_in"]
            .isel(
                **{
                    self.sdim_in: i,
                    self.tdim_in:  slice(0, self.T_in),
                    self.lev_in:   slice(self.lr[0],  self.lr[1]),
                    self.lat_in:   slice(self.latr[0], self.latr[1]),
                    self.lon_in:   slice(self.lonr[0], self.lonr[1]),
                }
            )
            .transpose(self.tdim_in, self.lev_in, self.lat_in, self.lon_in)
            .values
        )
        x_temp = _nan_clean(x_temp)

        # Y: temp_out -> (T_out, L, H, W)
        y_temp = (
            self._ds["temperatureNor_out"]
            .isel(
                **{
                    self.sdim_out: i,
                    self.tdim_out: slice(0, self.T_out),
                    self.lev_out:  slice(self.lr[0],  self.lr[1]),
                    self.lat_out:  slice(self.latr[0], self.latr[1]),
                    self.lon_out:  slice(self.lonr[0], self.lonr[1]),
                }
            )
            .transpose(self.tdim_out, self.lev_out, self.lat_out, self.lon_out)
            .values
        )
        y_temp = _nan_clean(y_temp)

        if self.needtauxy:
            # X: taux/tauy_in -> (T_in, H, W)
            x_taux = (
                self._ds["tauxNor_in"]
                .isel(**{
                    self.s_in: i,
                    self.t_in: slice(0, self.T_in),
                    self.lat_in_tau: slice(self.latr[0], self.latr[1]),
                    self.lon_in_tau: slice(self.lonr[0], self.lonr[1]),
                })
                .transpose(self.t_in, self.lat_in_tau, self.lon_in_tau)
                .values
            )
            x_tauy = (
                self._ds["tauyNor_in"]
                .isel(**{
                    self.s_in: i,
                    self.t_in: slice(0, self.T_in),
                    self.lat_in_tau: slice(self.latr[0], self.latr[1]),
                    self.lon_in_tau: slice(self.lonr[0], self.lonr[1]),
                })
                .transpose(self.t_in, self.lat_in_tau, self.lon_in_tau)
                .values
            )
            x_taux = _nan_clean(x_taux)
            x_tauy = _nan_clean(x_tauy)

            # Y: taux/tauy_out -> (T_out, H, W)
            y_taux = (
                self._ds["tauxNor_out"]
                .isel(**{
                    self.s_out: i,
                    self.t_out: slice(0, self.T_out),
                    self.lat_out_tau: slice(self.latr[0], self.latr[1]),
                    self.lon_out_tau: slice(self.lonr[0], self.lonr[1]),
                })
                .transpose(self.t_out, self.lat_out_tau, self.lon_out_tau)
                .values
            )
            y_tauy = (
                self._ds["tauyNor_out"]
                .isel(**{
                    self.s_out: i,
                    self.t_out: slice(0, self.T_out),
                    self.lat_out_tau: slice(self.latr[0], self.latr[1]),
                    self.lon_out_tau: slice(self.lonr[0], self.lonr[1]),
                })
                .transpose(self.t_out, self.lat_out_tau, self.lon_out_tau)
                .values
            )
            y_taux = _nan_clean(y_taux)
            y_tauy = _nan_clean(y_tauy)

            x = np.concatenate([x_taux[:, None], x_tauy[:, None], x_temp], axis=1).astype(np.float32)
            y = np.concatenate([y_taux[:, None], y_tauy[:, None], y_temp], axis=1).astype(np.float32)
        else:
            x, y = x_temp.astype(np.float32), y_temp.astype(np.float32)

        # 形状断言（调试一次即可）
        assert x.ndim == 4 and y.ndim == 4, f"Got x={x.shape}, y={y.shape}"
        return x, y


# ------------------ collator ------------------ #
@dataclass
class GeoformerCollator:
    to_float16: bool = False
    def __call__(self, batch: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, torch.Tensor]:
        x = np.stack([b[0] for b in batch], 0)  # (B,T_in,C,H,W)
        y = np.stack([b[1] for b in batch], 0)  # (B,T_out,C,H,W)
        dtype = torch.float16 if self.to_float16 else torch.float32
        return {
            "predictor": torch.from_numpy(x).to(dtype),
            "labels":    torch.from_numpy(y).to(dtype),
        }


class GeoformerFeatureDataset(Dataset):
    """读取预先导出的 Geoformer 特征 (`.npz`) 并返回 (history, future) 张量。"""

    def __init__(self, feature_files: Sequence[str | Path]):
        if not feature_files:
            raise ValueError("feature_files must not be empty")
        self.feature_files = [Path(p) for p in feature_files]
        for path in self.feature_files:
            if not path.exists():
                raise FileNotFoundError(f"Feature file not found: {path}")

    def __len__(self) -> int:
        return len(self.feature_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        path = self.feature_files[idx]
        with np.load(path, allow_pickle=False) as data:
            history = data["history"].astype(np.float32)
            future = data["future"].astype(np.float32)

        if history.ndim != 4 or future.ndim != 4:
            raise ValueError(
                f"Invalid feature shape in {path}: history {history.shape}, future {future.shape}"
            )
        return history, future


def list_geoformer_feature_files(root: str | Path, suffix: str = ".npz") -> List[Path]:
    root_path = Path(root)
    if root_path.is_file():
        return [root_path]
    if not root_path.exists():
        raise FileNotFoundError(f"Feature directory not found: {root_path}")
    return sorted(p for p in root_path.rglob(f"*{suffix}") if p.is_file())
