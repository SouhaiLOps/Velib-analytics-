# scripts/fetch_data.py
from __future__ import annotations

import os
import time
import datetime as dt
from pathlib import Path
from typing import Dict, Any, Optional, List

import requests
import pandas as pd
import numpy as np

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

# -------- Configuration --------
VELIB_BASE = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole"

# Permet de changer le dossier de sortie depuis l'environnement (GH Actions)
OUT_DIR = Path(os.environ.get("OUT_DIR", "data/raw/velib"))
OUT_DIR_SNAP = OUT_DIR / "snapshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR_SNAP.mkdir(parents=True, exist_ok=True)

STATION_INFO_CSV = OUT_DIR / "station_information.csv"
SYSTEM_INFO_JSON = OUT_DIR / "system_information.json"

PARQUET_COMPRESSION = os.environ.get("PARQUET_COMPRESSION", "snappy")  # snappy | gzip | zstd

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "velib-analytics/1.0 (+github actions)"})


# -------- Helpers --------
def fetch_json(url: str, retries: int = 3, timeout: int = 25) -> Dict[str, Any]:
    """Télécharge un JSON avec retries simples."""
    last_err: Optional[Exception] = None
    for i in range(retries):
        try:
            r = SESSION.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(1 + i)  # backoff simple
    raise RuntimeError(f"Failed to fetch {url}: {last_err!r}")


def get_system_information(base: str = VELIB_BASE) -> Dict[str, Any]:
    data = fetch_json(f"{base}/system_information.json")
    meta = {k: data.get(k) for k in ("lastUpdatedOther", "ttl") if k in data}
    sysinfo = data.get("data", {})
    return {"data": sysinfo, "meta": meta}


def get_gbfs_feeds(base: str = VELIB_BASE) -> Dict[str, Any]:
    data = fetch_json(f"{base}/gbfs.json")
    meta = {k: data.get(k) for k in ("lastUpdatedOther", "ttl") if k in data}
    feeds = data.get("data", {}).get("en", {}).get("feeds", [])
    return {"data": feeds, "meta": meta}


def get_station_information(base: str = VELIB_BASE) -> pd.DataFrame:
    data = fetch_json(f"{base}/station_information.json")
    stations = data.get("data", {}).get("stations", [])
    df = pd.DataFrame(stations)

    if "station_id" in df.columns:
        df["id"] = df["station_id"].astype(str)
    else:
        df["id"] = pd.NA

    for col in ["capacity", "lat", "lon", "name"]:
        if col not in df.columns:
            df[col] = pd.NA

    cols: List[str] = ["id", "name", "lat", "lon", "capacity"]
    if "station_id" in df.columns:
        cols.insert(1, "station_id")
    df = df[cols]

    # sauvegarde informative (overwrite)
    df.to_csv(STATION_INFO_CSV, index=False)
    return df


def get_station_status(base: str = VELIB_BASE) -> pd.DataFrame:
    data = fetch_json(f"{base}/station_status.json")
    stations = data.get("data", {}).get("stations", [])
    df = pd.DataFrame(stations)

    if "station_id" in df.columns:
        df["id"] = df["station_id"].astype(str)
    else:
        df["id"] = pd.NA

    def pick_col(row, names):
        for n in names:
            if n in row and pd.notna(row[n]):
                return row[n]
        return np.nan

    df["num_bikes_available_norm"] = df.apply(
        lambda r: pick_col(r, ["num_bikes_available", "numBikesAvailable"]), axis=1
    )
    df["num_docks_available_norm"] = df.apply(
        lambda r: pick_col(r, ["num_docks_available", "numDocksAvailable"]), axis=1
    )

    mech, elec = [], []
    nba_types = df.get("num_bikes_available_types")
    if nba_types is not None:
        for item in nba_types:
            m = e = 0
            if isinstance(item, list):
                for d in item:
                    if isinstance(d, dict):
                        m += int(d.get("mechanical", 0) or 0)
                        e += int(d.get("ebike", 0) or 0)
            elif isinstance(item, dict):
                m = int(item.get("mechanical", 0) or 0)
                e = int(item.get("ebike", 0) or 0)
            mech.append(m)
            elec.append(e)
    else:
        mech = [np.nan] * len(df)
        elec = [np.nan] * len(df)

    df["available_mechanical"] = mech
    df["available_ebike"] = elec

    for col in ["is_installed", "is_renting", "is_returning"]:
        if col not in df.columns:
            df[col] = np.nan

    if "last_reported" in df.columns:
        df["last_reported"] = pd.to_datetime(
            df["last_reported"], unit="s", errors="coerce", utc=True
        )

    return df


def build_snapshot_df(base: str = VELIB_BASE, tzname: str = "UTC") -> pd.DataFrame:
    info = get_station_information(base)
    status = get_station_status(base)

    merged = status.merge(
        info[["id", "name", "lat", "lon", "capacity"]],
        on="id",
        how="left",
    )

    now_utc = pd.to_datetime(dt.datetime.now(dt.timezone.utc), utc=True)
    try:
        if ZoneInfo and tzname:
            now_local = now_utc.tz_convert(ZoneInfo(tzname))
        else:
            now_local = now_utc
    except Exception:
        now_local = now_utc

    merged["ts_utc"] = now_utc
    merged["ts_local"] = now_local

    cap = pd.to_numeric(merged.get("capacity"), errors="coerce")
    bikes = pd.to_numeric(merged.get("num_bikes_available_norm"), errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        merged["fill_rate"] = np.where(cap > 0, bikes / cap, np.nan)

    preferred = [
        "ts_utc", "ts_local", "id", "name", "lat", "lon", "capacity",
        "num_bikes_available_norm", "num_docks_available_norm",
        "available_mechanical", "available_ebike",
        "is_installed", "is_renting", "is_returning", "last_reported",
        "fill_rate",
    ]
    for c in preferred:
        if c not in merged.columns:
            merged[c] = np.nan

    merged = merged[preferred + [c for c in merged.columns if c not in preferred]]
    return merged


def save_snapshot_parquet(df: pd.DataFrame, out_dir: Path = OUT_DIR_SNAP) -> Path:
    ts = pd.to_datetime(df["ts_utc"].iloc[0])
    stamp = ts.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"velib_snapshot_{stamp}.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow", compression=PARQUET_COMPRESSION)
    return out_path


def main() -> None:
    # system info (name / tz) + sauvegarde JSON
    sysinfo = get_system_information(VELIB_BASE)
    tzname = sysinfo["data"].get("timezone", "UTC")
    (SYSTEM_INFO_JSON).write_text(pd.Series(sysinfo).to_json(), encoding="utf-8")

    df = build_snapshot_df(VELIB_BASE, tzname=tzname)
    out_path = save_snapshot_parquet(df)
    print(f"[OK] snapshot stations={len(df)} -> {out_path}")


if __name__ == "__main__":
    main()



