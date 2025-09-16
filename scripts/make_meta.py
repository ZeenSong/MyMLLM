import xarray as xr
import json
import os

def make_meta(nc_path, out_dir="data/raw"):
    os.makedirs(out_dir, exist_ok=True)

    # 1. 打开原始 NetCDF 文件 (只读)
    ds = xr.open_dataset(nc_path)

    # 2. 我们关心的变量分组
    inputs = ["temperatureNor", "tauxNor", "tauyNor"]
    diagnostics = ["nino34", "stdtemp", "stdtaux", "stdtauy"]

    # 3. 读取坐标
    coords = {}
    for c in ["lev", "lat", "lon"]:
        if c in ds.coords:
            coords[c] = ds[c].values.tolist()
    coords["n_model"] = ds.dims.get("n_model", None)
    coords["n_mon"] = ds.dims.get("n_mon", None)

    # 4. 组织 meta 信息
    meta = {
        "source_file": nc_path,
        "variables": {
            "inputs": inputs,
            "diagnostics": diagnostics
        },
        "coords": coords,
        "fill_value_rule": "NaN -> 0; |x| > 999 -> 0",
        "note": "原始nc文件不修改，只在Dataset里根据规则做处理"
    }

    # 5. 保存 meta.json
    out_meta = os.path.join(out_dir, "meta.json")
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[S0] meta.json 已保存到 {out_meta}")


if __name__ == "__main__":
    nc_file = "dataset/3DGeoformer_origin_data/CMIP6_separate_model_up150m_tauxy_Nor_kb.nc"
    make_meta(nc_file)