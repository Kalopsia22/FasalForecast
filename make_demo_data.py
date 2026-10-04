"""
Generate DEMO inputs so the pipeline and app can be tried without Earth Engine.
Yields are REAL (ICRISAT); satellite/weather features are SYNTHETIC, built to correlate
with yield. Any accuracy shown on this data is meaningless. Writes to data_demo/ only.
"""
import json, os
import numpy as np
import pandas as pd
from common import CROPS, N_MONTHS, GAUL_STATE_ALIAS

rng = np.random.default_rng(42)
os.makedirs("data_demo", exist_ok=True)
yl = pd.read_csv("data/yield_long.csv")
keys = yl[["state", "district"]].drop_duplicates().sort_values(["state", "district"]).reset_index(drop=True)
keys["ADM2_CODE"] = 10000 + keys.index
rev = {v: k for k, v in GAUL_STATE_ALIAS.items()}

NDVI = {11: [.25, .40, .60, .75, .55, .35], 10: [.30, .50, .65, .70, .45, .30], 6: [.30, .50, .65, .70, .55, .35]}
TEMP = {11: [22, 18, 16, 19, 24, 29], 10: [25, 21, 18, 16, 19, 24], 6: [32, 29, 27, 26, 25, 22]}
RAIN = {11: [5, 5, 15, 15, 10, 10], 10: [20, 5, 5, 15, 15, 10], 6: [100, 250, 300, 220, 100, 30]}


def synth(crop, df, q, partial=False):
    sm = CROPS[crop]["start_month"]
    n = len(df)
    out = df[["state", "district", "ADM2_CODE", "year"]].copy()
    out["ADM1_NAME"] = out.state.replace(rev)
    out["ADM2_NAME"] = out.district
    for i in range(N_MONTHS):
        out[f"ndvi_s{i+1}"] = np.clip(NDVI[sm][i] * (1 + .12 * q) + .03 * rng.normal(size=n), .05, .95)
        out[f"rain_s{i+1}"] = np.clip(RAIN[sm][i] * (1 + .25 * q) + 3 * rng.normal(size=n), 0, None)
        out[f"tmean_s{i+1}"] = TEMP[sm][i] - .8 * q + .5 * rng.normal(size=n)
    if partial:                                   # pretend the season is only 4 months old
        for i in (5, 6):
            for v in ("ndvi", "rain", "tmean"):
                out[f"{v}_s{i}"] = np.nan
    return out.drop(columns=["state", "district"])


for crop in CROPS:
    s = yl[yl.crop == crop].merge(keys, on=["state", "district"]).copy()
    z = s.groupby(["state", "district"]).yield_kgha.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    hist = synth(crop, s, .8 * z.values + .6 * rng.normal(size=len(s)))
    cur = s[s.year == s.year.max()].copy()
    cur["year"] = s.year.max() + 1                # a "current" season with no yield yet
    cur = synth(crop, cur, rng.normal(size=len(cur)), partial=True)
    pd.concat([hist, cur]).to_csv(f"data_demo/gee_features_{crop}.csv", index=False)

feats = []
for i, r in keys.iterrows():
    x0, y0 = 68 + (i % 30) * .55, 8 + (i // 30) * .5
    ring = [[x0, y0], [x0 + .5, y0], [x0 + .5, y0 + .45], [x0, y0 + .45], [x0, y0]]
    feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                  "properties": {"ADM2_CODE": int(r.ADM2_CODE), "ADM1_NAME": rev.get(r.state, r.state),
                                 "ADM2_NAME": r.district}})
json.dump({"type": "FeatureCollection", "features": feats}, open("data_demo/districts.geojson", "w"))
open("data_demo/DEMO_DATA", "w").write("synthetic features - demo only")
print("demo data written to data_demo/")
