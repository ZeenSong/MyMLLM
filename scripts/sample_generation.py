import os
import json
import numpy as np
import xarray as xr
import argparse

def make_samples(nc_path, meta_path, out_dir="data/samples",
                 input_length=12, output_length=20,
                 n_samples=1000, needtauxy=True):
    os.makedirs(out_dir, exist_ok=True)

    # 1. 读取 meta.json
    with open(meta_path, "r") as f:
        meta = json.load(f)
    input_vars = meta["variables"]["inputs"]

    # 2. 打开原始 nc 文件
    ds = xr.open_dataset(nc_path)

    # 3. 提取核心变量
    temp = ds["temperatureNor"].values  # (n_model, n_mon, lev, lat, lon)
    taux = ds["tauxNor"].values         # (n_model, n_mon, lat, lon)
    tauy = ds["tauyNor"].values

    # 缺测值处理
    def clean(x):
        x = np.nan_to_num(x)
        x[np.abs(x) > 999] = 0
        return x
    temp, taux, tauy = clean(temp), clean(taux), clean(tauy)

    # 4. 组合成统一 field_data
    # shape = (n_model, n_mon, C, lat, lon), C = 2 + n_lev
    field_data = np.concatenate(
        (taux[:, :, None], tauy[:, :, None], temp),
        axis=2
    )

    n_model, n_mon, C, H, W = field_data.shape
    st_min = input_length - 1
    ed_max = n_mon - output_length - 1

    print(f"[INFO] field_data shape = {field_data.shape}")
    print(f"[INFO] 可采样时间范围: {st_min} ~ {ed_max}")

    # 5. 随机采样
    for i in range(n_samples):
        rd_m = np.random.randint(0, n_model)
        rd_t = np.random.randint(st_min, ed_max)

        x = field_data[rd_m, rd_t - input_length + 1 : rd_t + 1]   # [T_in, C, H, W]
        y = field_data[rd_m, rd_t + 1 : rd_t + 1 + output_length]  # [T_out, C, H, W]

        out_path = os.path.join(out_dir, f"sample_{i:06d}.npz")
        np.savez_compressed(out_path, x=x.astype(np.float32), y=y.astype(np.float32))

        if i % 100 == 0:
            print(f"[S1] Saved {out_path}")

    ds.close()
    print(f"[DONE] 共生成 {n_samples} 个样本，保存到 {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc_path", type=str,
                        default="dataset/3DGeoformer_origin_data/CMIP6_separate_model_up150m_tauxy_Nor_kb.nc")
    parser.add_argument("--meta_path", type=str, default="data/raw/meta.json")
    parser.add_argument("--out_dir", type=str, default="data/samples")
    parser.add_argument("--input_length", type=int, default=12)
    parser.add_argument("--output_length", type=int, default=20)
    parser.add_argument("--n_samples", type=int, default=1000)
    args = parser.parse_args()

    make_samples(args.nc_path, args.meta_path,
                 out_dir=args.out_dir,
                 input_length=args.input_length,
                 output_length=args.output_length,
                 n_samples=args.n_samples)
