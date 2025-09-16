"""Utilities to sample Geoformer data and build automatic SFT annotations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import xarray as xr


def _nan_clean(a: np.ndarray) -> np.ndarray:
    cleaned = np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    cleaned[np.abs(cleaned) > 999] = 0.0
    return cleaned


def _basic_stats(arr: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _trend_desc(delta: float, threshold: float, up: str, down: str, steady: str) -> str:
    if delta > threshold:
        return up
    if delta < -threshold:
        return down
    return steady


@dataclass
class SFTSamplerConfig:
    dataset_path: Path
    min_input_length: int
    max_input_length: int
    min_output_length: int
    max_output_length: int
    num_samples: int
    seed: int = 0


@dataclass
class SFTStats:
    temperature: Dict[str, float]
    surface_mean: float
    deep_mean: float
    taux: Dict[str, float]
    tauy: Dict[str, float]
    nino: Dict[str, float]
    nino_trend: float


@dataclass
class SFTItem:
    sample_id: str
    task: str  # "analysis" or "prediction"
    prompt: str
    response: str
    history: np.ndarray
    future: np.ndarray
    metadata: Dict[str, object]


class GeoformerSFTSampler:
    def __init__(self, config: SFTSamplerConfig):
        self.config = config

    def _extract_temperature(self, ds: xr.Dataset, model_idx: int, start_month: int, length: int) -> np.ndarray:
        window = ds["temperatureNor"].isel(
            n_model=model_idx,
            n_mon=slice(start_month, start_month + length),
        )
        return _nan_clean(window.to_numpy())

    def _extract_wind(self, ds: xr.Dataset, var: str, model_idx: int, start_month: int, length: int) -> np.ndarray:
        window = ds[var].isel(
            n_model=model_idx,
            n_mon=slice(start_month, start_month + length),
        )
        return _nan_clean(window.to_numpy())

    def _extract_nino(self, ds: xr.Dataset, model_idx: int, start_month: int, length: int) -> np.ndarray:
        window = ds["nino34"].isel(
            n_model=model_idx,
            n_mon=slice(start_month, start_month + length),
        )
        return _nan_clean(window.to_numpy())

    def _summarise_temperature(self, temp: np.ndarray) -> Tuple[Dict[str, float], float, float]:
        stats = _basic_stats(temp)
        surface_mean = float(temp[:, 0].mean()) if temp.shape[1] > 0 else float(temp.mean())
        deep_mean = float(temp[:, -1].mean()) if temp.shape[1] > 0 else float(temp.mean())
        return stats, surface_mean, deep_mean

    def _summarise_wind(self, wind: np.ndarray) -> Dict[str, float]:
        return _basic_stats(wind)

    def _summarise_nino(self, nino: np.ndarray) -> Tuple[Dict[str, float], float]:
        stats = _basic_stats(nino)
        trend = float(nino[-1] - nino[0]) if len(nino) > 1 else 0.0
        return stats, trend

    def _build_prediction_prompt(
        self,
        hist_stats: SFTStats,
        history_length: int,
        future_len: int,
    ) -> str:
        return (
            f"<geoformer>请基于提供的海洋历史序列（持续 {history_length} 个月）"
            f"预测未来 {future_len} 个月的海温演变与厄尔尼诺发展趋势，并给出要点分析。"
        )

    def _build_prediction_response(self, hist: SFTStats, future: SFTStats, future_len: int) -> str:
        temp_diff = future.temperature["mean"] - hist.temperature["mean"]
        temp_trend = _trend_desc(
            temp_diff,
            threshold=0.05,
            up="整体偏暖",
            down="整体转凉",
            steady="与过去相近",
        )
        nino_trend = _trend_desc(
            future.nino_trend,
            threshold=0.1,
            up="将继续升高",
            down="可能回落",
            steady="大体平稳",
        )
        wind_delta = future.taux["mean"] - hist.taux["mean"]
        wind_desc = _trend_desc(
            wind_delta,
            threshold=0.02,
            up="东向风应力有所增强",
            down="东向风应力有所减弱",
            steady="东向风保持稳定",
        )
        return (
            f"未来 {future_len} 个月的平均海温约为 {future.temperature['mean']:.2f}°C，"
            f"{temp_trend}，最高 {future.temperature['max']:.2f}°C，最低 {future.temperature['min']:.2f}°C。"
            f"表层平均 {future.surface_mean:.2f}°C，深层平均 {future.deep_mean:.2f}°C。"
            f"Nino3.4 指数均值 {future.nino['mean']:.2f}，标准差 {future.nino['std']:.2f}，"
            f"峰值 {future.nino['max']:.2f}，谷值 {future.nino['min']:.2f}，整体来看 {nino_trend}。"
            f"风应力方面，东向平均 {future.taux['mean']:.3f}，南向平均 {future.tauy['mean']:.3f}，{wind_desc}。"
        )

    def _build_analysis_prompt(
        self,
        hist_stats: SFTStats,
        history_length: int,
    ) -> str:
        return (
            f"<geoformer>请基于提供的海洋历史序列（持续 {history_length} 个月）总结主要的海洋-大气特征"
            "，包括海温层结、风应力与 Nino3.4 指数的变化，并指出值得关注的气候信号。"
        )

    def _build_analysis_response(self, hist_stats: SFTStats) -> str:
        temp_desc = _trend_desc(
            hist_stats.temperature["mean"],
            threshold=0.1,
            up="海温偏暖",
            down="海温偏冷",
            steady="海温接近常态",
        )
        nino_trend = _trend_desc(
            hist_stats.nino_trend,
            threshold=0.1,
            up="显示厄尔尼诺增强迹象",
            down="呈现拉尼娜倾向",
            steady="暂时缺乏明显的厄尔尼诺/拉尼娜信号",
        )
        wind_state = _trend_desc(
            hist_stats.taux["mean"],
            threshold=0.02,
            up="盛行偏强的西风异常",
            down="东风异常显著",
            steady="风场总体平衡",
        )
        return (
            f"历史海温均值约 {hist_stats.temperature['mean']:.2f}°C，{temp_desc}，"
            f"垂直方向表层 {hist_stats.surface_mean:.2f}°C、深层 {hist_stats.deep_mean:.2f}°C，" 
            "显示上下层温差结构。"
            f"Nino3.4 指数波动范围 [{hist_stats.nino['min']:.2f}, {hist_stats.nino['max']:.2f}]，"
            f"均值 {hist_stats.nino['mean']:.2f}，{nino_trend}。"
            f"风应力均值（东向 {hist_stats.taux['mean']:.3f}，南向 {hist_stats.tauy['mean']:.3f}），{wind_state}，"
            "建议关注对应海区的海气耦合及潜在的强对流发展。"
        )

    def _build_metadata(
        self,
        model_idx: int,
        hist_stats: SFTStats,
        future_stats: SFTStats,
        history_start: int,
        history_end: int,
        future_end: int,
        history_length: int,
        future_length: int,
        task: str,
    ) -> Dict[str, object]:
        return {
            "model_index": model_idx,
            "history_start": history_start,
            "history_end": history_end,
            "future_start": history_end + 1,
            "future_end": future_end,
            "history_length": history_length,
            "future_length": future_length,
            "task": task,
            "history": {
                "temperature": hist_stats.temperature,
                "surface_mean": hist_stats.surface_mean,
                "deep_mean": hist_stats.deep_mean,
                "taux": hist_stats.taux,
                "tauy": hist_stats.tauy,
                "nino": hist_stats.nino,
                "nino_trend": hist_stats.nino_trend,
            },
            "future": {
                "temperature": future_stats.temperature,
                "surface_mean": future_stats.surface_mean,
                "deep_mean": future_stats.deep_mean,
                "taux": future_stats.taux,
                "tauy": future_stats.tauy,
                "nino": future_stats.nino,
                "nino_trend": future_stats.nino_trend,
            },
        }

    def _sample_once(self, ds: xr.Dataset, rng: np.random.Generator, idx: int) -> SFTItem:
        max_month = ds.sizes["n_mon"]
        model_idx = int(rng.integers(0, ds.sizes["n_model"]))

        hist_len = int(rng.integers(self.config.min_input_length, self.config.max_input_length + 1))
        fut_len = int(rng.integers(self.config.min_output_length, self.config.max_output_length + 1))

        last_hist = int(rng.integers(hist_len - 1, max_month - fut_len - 1))
        hist_start = last_hist - hist_len + 1
        future_start = last_hist + 1
        future_end = future_start + fut_len - 1

        hist_temp = self._extract_temperature(ds, model_idx, hist_start, hist_len)
        fut_temp = self._extract_temperature(ds, model_idx, future_start, fut_len)
        hist_taux = self._extract_wind(ds, "tauxNor", model_idx, hist_start, hist_len)
        fut_taux = self._extract_wind(ds, "tauxNor", model_idx, future_start, fut_len)
        hist_tauy = self._extract_wind(ds, "tauyNor", model_idx, hist_start, hist_len)
        fut_tauy = self._extract_wind(ds, "tauyNor", model_idx, future_start, fut_len)
        hist_nino = self._extract_nino(ds, model_idx, hist_start, hist_len)
        fut_nino = self._extract_nino(ds, model_idx, future_start, fut_len)

        hist_temp_stats, hist_surface, hist_deep = self._summarise_temperature(hist_temp)
        fut_temp_stats, fut_surface, fut_deep = self._summarise_temperature(fut_temp)
        hist_taux_stats = self._summarise_wind(hist_taux)
        fut_taux_stats = self._summarise_wind(fut_taux)
        hist_tauy_stats = self._summarise_wind(hist_tauy)
        fut_tauy_stats = self._summarise_wind(fut_tauy)
        hist_nino_stats, hist_nino_trend = self._summarise_nino(hist_nino)
        fut_nino_stats, fut_nino_trend = self._summarise_nino(fut_nino)

        hist_stats = SFTStats(
            temperature=hist_temp_stats,
            surface_mean=hist_surface,
            deep_mean=hist_deep,
            taux=hist_taux_stats,
            tauy=hist_tauy_stats,
            nino=hist_nino_stats,
            nino_trend=hist_nino_trend,
        )
        fut_stats = SFTStats(
            temperature=fut_temp_stats,
            surface_mean=fut_surface,
            deep_mean=fut_deep,
            taux=fut_taux_stats,
            tauy=fut_tauy_stats,
            nino=fut_nino_stats,
            nino_trend=fut_nino_trend,
        )

        pred_prompt = self._build_prediction_prompt(hist_stats, hist_len, fut_len)
        pred_response = self._build_prediction_response(hist_stats, fut_stats, fut_len)
        ana_prompt = self._build_analysis_prompt(hist_stats, hist_len)
        ana_response = self._build_analysis_response(hist_stats)
        task = "prediction" if rng.random() < 0.5 else "analysis"
        prompt = pred_prompt if task == "prediction" else ana_prompt
        response = pred_response if task == "prediction" else ana_response

        metadata = self._build_metadata(
            model_idx,
            hist_stats,
            fut_stats,
            hist_start,
            last_hist,
            future_end,
            hist_len,
            fut_len,
            task,
        )

        history_tensor = np.concatenate(
            [hist_taux[:, None], hist_tauy[:, None], hist_temp], axis=1
        )
        future_tensor = np.concatenate(
            [fut_taux[:, None], fut_tauy[:, None], fut_temp], axis=1
        )

        sample_id = f"sft-{model_idx}-{last_hist}-{idx}-{task}"
        return SFTItem(
            sample_id=sample_id,
            task=task,
            prompt=prompt,
            response=response,
            history=history_tensor,
            future=future_tensor,
            metadata=metadata,
        )

    def sample(self) -> Iterator[SFTItem]:
        rng = np.random.default_rng(self.config.seed)
        with xr.open_dataset(self.config.dataset_path) as ds:
            for idx in range(self.config.num_samples):
                yield self._sample_once(ds, rng, idx)


def build_samples(config: SFTSamplerConfig) -> List[SFTItem]:
    sampler = GeoformerSFTSampler(config)
    return list(sampler.sample())
