"""Turn the raw ICRISAT file into a tidy long yield table for all crops.
Usage: python 00_prepare_yield.py [--raw ../icrisat_district_level_raw.csv]
Writes data/yield_long.csv  (crop, state, district, year, area_kha, yield_kgha)"""
import argparse
import pandas as pd
from common import CROPS

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="../icrisat_district_level_raw.csv")
ap.add_argument("--min-year", type=int, default=2000)
ap.add_argument("--min-area-kha", type=float, default=0.5,
                help="drop district-years with tiny area (500 ha); yields there are unreliable")
a = ap.parse_args()

raw = pd.read_csv(a.raw)
frames = []
for crop, cfg in CROPS.items():
    k = cfg["icrisat"]
    d = raw[["State Name", "Dist Name", "Year",
             f"{k} AREA (1000 ha)", f"{k} YIELD (Kg per ha)"]].copy()
    d.columns = ["state", "district", "year", "area_kha", "yield_kgha"]
    d["crop"] = crop
    frames.append(d)
out = pd.concat(frames)
n0 = len(out)
out = out[(out.year >= a.min_year) & (out.area_kha >= a.min_area_kha) & (out.yield_kgha > 0)]
cap = out.crop.map(lambda c: CROPS[c]["max_yield"])
print(f"dropped {int((out.yield_kgha > cap).sum())} implausible-yield rows (above per-crop cap)")
out = out[out.yield_kgha <= cap]
print(f"kept {len(out)} of {n0} rows")
out = out[["crop", "state", "district", "year", "area_kha", "yield_kgha"]]
out.to_csv("data/yield_long.csv", index=False)
print(out.groupby("crop").agg(rows=("year", "size"), districts=("district", "nunique")))
