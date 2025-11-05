import xarray as xr

LATEST = "1.1"


def save_dataset(ds: xr.Dataset, path: str, *, engine: str | None = None) -> None:
    ds = ds.copy()
    ds.attrs.setdefault("qualibrate_schema_version", LATEST)
    enc = {k: {"zlib": True, "complevel": 3} for k in ds.data_vars}
    ds.to_netcdf(path, engine=engine, encoding=enc)


def load_dataset(path: str, *, engine: str | None = None, target_version: str = LATEST) -> xr.Dataset:
    ds = xr.load_dataset(path, engine=engine)
    src = str(ds.attrs.get("qualibrate_schema_version", "1.0"))
    return upgrade(ds, src, target_version)


def upgrade(ds: xr.Dataset, src: str, dst: str) -> xr.Dataset:
    cur = src
    out = ds
    while cur != dst:
        if cur == "1.0" and dst in ("1.1",):
            out = _upgrade_10_to_11(out)
            cur = "1.1"
        else:
            out.attrs["compat_warning"] = f"Unrecognized upgrade path {cur}->{dst}"
            break
    return out


def _upgrade_10_to_11(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    for c in ds.coords:
        ds[c].attrs.setdefault("long_name", c)
    if "population" in ds.data_vars and "state" not in ds.data_vars:
        ds = ds.rename({"population": "state"})
    ds.attrs["qualibrate_schema_version"] = "1.1"
    return ds